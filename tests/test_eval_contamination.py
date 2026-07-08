from pathlib import Path
import json
import hashlib

from training.eval_v2 import PRIVATE_EVAL_SUITE, COMPACT_EVAL_SUITE
from runtime.experience_ledger import content_hash


def test_eval_prompts_not_in_training_shards(tmp_path: Path) -> None:
    # 1. Gather all hashes we want to protect
    protected_hashes = set()
    for suite in (PRIVATE_EVAL_SUITE, COMPACT_EVAL_SUITE):
        for task in suite:
            # Hash of the raw prompt text
            prompt_text = str(task["prompt"])
            protected_hashes.add(hashlib.sha256(prompt_text.encode("utf-8")).hexdigest())
            protected_hashes.add(content_hash(prompt_text))
            protected_hashes.add(content_hash({"prompt": prompt_text}))

    # 2. Mock a training shard directory
    train_dir = tmp_path / "train_shards"
    train_dir.mkdir()
    
    # 3. Create a clean shard
    clean_shard = train_dir / "shard_001.jsonl"
    clean_events = [
        {"inputs_hash": content_hash("hello"), "output": "world"},
        {"inputs_hash": content_hash("test"), "output": "success"}
    ]
    with clean_shard.open("w") as f:
        for event in clean_events:
            f.write(json.dumps(event) + "\n")

    # 4. Create a contaminated shard (mocking a leak)
    leak_prompt = str(PRIVATE_EVAL_SUITE[0]["prompt"])
    leak_hash = content_hash(leak_prompt)
    contam_shard = train_dir / "shard_002.jsonl"
    contam_events = [
        {"inputs_hash": leak_hash, "output": "leaked"}
    ]
    with contam_shard.open("w") as f:
        for event in contam_events:
            f.write(json.dumps(event) + "\n")
            
    # 5. Define the firewall check function
    def check_contamination(shard_dir: Path, protected: set[str]) -> list[str]:
        violations = []
        for path in shard_dir.glob("*.jsonl"):
            with path.open("r") as f:
                for line_idx, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        # Check both exact prompt hashes and inputs_hash
                        if event.get("inputs_hash") in protected:
                            violations.append(f"{path.name}:{line_idx} contains protected hash {event['inputs_hash']}")
                        # If there's raw text, we could hash it and check, but typical training shards
                        # only store hashes of the inputs for exact-match deduplication or lookup.
                    except json.JSONDecodeError:
                        violations.append(f"{path.name}:{line_idx} is malformed JSON, potential obfuscated leak")
        return violations

    # 6. Verify it catches the leak
    violations = check_contamination(train_dir, protected_hashes)
    assert len(violations) == 1
    assert "shard_002.jsonl:1" in violations[0]
