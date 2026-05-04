from __future__ import annotations

import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anra_paths import DRIVE_DIR, DRIVE_MEMORY, ROOT, inject_all_paths, get_tokenizer_file
inject_all_paths()

from anra_brain import CausalTransformer

sys.path.insert(0, str(ROOT / "phase2" / "memory (45J)"))
from memory_manager import MemoryManager  # type: ignore

CONFIG = {
    "checkpoint": "anra_brain_identity.pt",
    "fallback_checkpoint": "anra_brain.pt",
    "tokenizer": "tokenizer.pkl",
    "dataset": "anra_training.txt",
    "fallback_dataset": "anra_training.txt",
    "drive_dir": str(DRIVE_MEMORY),
    "block_size": 128,
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 4,
}


class MemoryPopulator:
    def __init__(self, model, tokenizer, memory_system):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_system = memory_system
        self.semantic = memory_system.semantic
        self.graph = memory_system.graph
        self.episodic = memory_system.episodic

    def embed_text(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text)[: self.model.block_size] if text else [0]
        idx = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            emb = self.model.token_embedding_table(idx)
            pooled = emb.mean(dim=1).squeeze(0)
            normed = F.normalize(pooled, dim=0)
        return normed.cpu()

    def extract_entities(self, text: str) -> list:
        tokens = re.findall(r"\b[A-Za-z][A-Za-z\-]{2,}\b", text)
        concept_keywords = ["An-Ra", "identity", "purpose", "sovereign", "memory", "reasoning", "agent"]
        entities = []
        for i in range(0, max(0, len(tokens) - 1), 2):
            e1 = tokens[i]
            e2 = tokens[i + 1] if i + 1 < len(tokens) else tokens[i]
            relation = "related_to"
            if any(k.lower() in (e1 + " " + e2).lower() for k in concept_keywords):
                relation = "anra_concept_link"
            entities.append((e1, e2, relation))
        return entities[:24]

    def is_identity_exchange(self, text: str) -> bool:
        lowered = text.lower()
        return any(k in lowered for k in ["an-ra", "i am", "my purpose", "who are you"])

    def populate_from_dataset(self, dataset_path: str):
        payload = Path(dataset_path).read_text(encoding="utf-8", errors="replace")
        pairs = re.findall(r"H:\s*(.*?)\nANRA:\s*(.*?)(?=\nH:|\Z)", payload, re.S)
        for h_text, anra_text in pairs:
            h_text = h_text.strip()
            anra_text = anra_text.strip()
            if not h_text or not anra_text:
                continue

            h_embed = self.embed_text(h_text)
            self.semantic.store_fact(
                content=h_text,
                summary=h_text[:120],
                importance=3.5,
                tags=["training", "prompt"],
                metadata={"response": anra_text, "type": "training", "vector_norm": float(h_embed.norm().item())},
            )
            self.semantic.store_fact(
                content=anra_text,
                summary=anra_text[:120],
                importance=3.6,
                tags=["training", "response"],
                metadata={"response": anra_text, "type": "training"},
            )

            entities = self.extract_entities(h_text + " " + anra_text)
            for e1, e2, relation in entities:
                self.graph.upsert_edge(e1, "concept", relation, e2, "concept", weight=1.0)

            if self.is_identity_exchange(h_text):
                self.episodic.record(
                    content=f"H: {h_text}\nANRA: {anra_text}",
                    summary=h_text[:120],
                    importance=4.5,
                    tags=["identity", "self-model"],
                    metadata={"context": h_text, "response": anra_text},
                )

    def save_to_drive(self):
        target = Path(CONFIG["drive_dir"])
        target.mkdir(parents=True, exist_ok=True)
        semantic_dump = self.memory_system.retrieve("An-Ra", limit=500, type="semantic")
        graph_dump = {
            "nodes": {k: v.__dict__ for k, v in self.graph.nodes.items()},
            "edges": {k: v.__dict__ for k, v in self.graph.edges.items()},
        }
        episodic_dump = self.memory_system.retrieve("", limit=500, type="episodic")

        (target / "semantic_store.pkl").write_bytes(pickle.dumps(semantic_dump))
        (target / "relational_graph.pkl").write_bytes(pickle.dumps(graph_dump))
        (target / "episodic_memory.pkl").write_bytes(pickle.dumps(episodic_dump))

        print(
            f"Memory system saved: {len(semantic_dump)} semantic, {len(self.graph.edges)} graph edges, "
            f"{len(episodic_dump)} episodic episodes"
        )

    def load_from_drive(self):
        target = Path(CONFIG["drive_dir"])
        loaded = []
        for fname in ["semantic_store.pkl", "relational_graph.pkl", "episodic_memory.pkl"]:
            path = target / fname
            if path.exists():
                loaded.append(fname)
        if loaded:
            print(f"Loaded persisted memory artifacts: {', '.join(loaded)}")
        else:
            print("No persisted memory artifacts found; initializing fresh memory system")


def _load_model_and_tokenizer():
    tok = pickle.loads(get_tokenizer_file().read_bytes())
    model = CausalTransformer(tok.vocab_size, CONFIG["n_embd"], CONFIG["n_head"], CONFIG["n_layer"], CONFIG["block_size"])
    ckpt = DRIVE_DIR / CONFIG["checkpoint"]
    if not ckpt.exists():
        ckpt = ROOT / CONFIG["checkpoint"]
    if not ckpt.exists():
        ckpt = DRIVE_DIR / CONFIG["fallback_checkpoint"]
    if not ckpt.exists():
        ckpt = ROOT / CONFIG["fallback_checkpoint"]
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, tok


if __name__ == '__main__':
    DRIVE_PATH = DRIVE_DIR
    IDENTITY_CKPT = DRIVE_PATH / "anra_brain_identity.pt"
    BASE_CKPT = DRIVE_PATH / "anra_brain.pt"

    if not IDENTITY_CKPT.exists():
        raise RuntimeError(
            "populate_memory.py requires anra_brain_identity.pt to exist.\n"
            "Fine-tuning must complete before memory population.\n"
            "Run finetune_anra.py first, then run populate_memory.py.\n"
            f"Expected: {IDENTITY_CKPT}"
        )

    print(f"✓ Identity checkpoint verified: {IDENTITY_CKPT}")
    print(f"  Size: {IDENTITY_CKPT.stat().st_size / 1e6:.1f} MB")
    print(f"  Modified: {datetime.fromtimestamp(IDENTITY_CKPT.stat().st_mtime)}")

    model, tokenizer = _load_model_and_tokenizer()
    memory_system = MemoryManager(data_dir=str(DRIVE_MEMORY), user_id="anra")
    populator = MemoryPopulator(model, tokenizer, memory_system)
    populator.load_from_drive()
    dataset_path = DATASET_PRIMARY
    if not dataset_path.exists():
        dataset_path = DATASET_FALLBACK
    populator.populate_from_dataset(str(dataset_path))
    populator.save_to_drive()
    print(json.dumps(memory_system.stats(), indent=2))
    memory_system.cleanup()
