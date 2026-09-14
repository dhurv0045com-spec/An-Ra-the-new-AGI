"""CS-TRANSFER-001 executable revision v4 (Amendment 2).

The first real prepare pass proved the preregistered natural-language low-ID
surface infeasible under the frozen tokenizer (sealed/identity acceptance 0.000)
before any model update. Amendment 2 replaces only that data instrument with a
deterministic direct-token surface entirely inside IDs 260..508. The physical
V4096-vs-V24576 treatment, models, initialization, optimizer, schedule, fixed
endpoint, evaluation checkpoints, decision thresholds, and sealed firewall are
unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json

from anra_v5 import cs_transfer_001_data_v2 as data_v2
from anra_v5 import cs_transfer_001_run as r
from anra_v5 import cs_transfer_001_run_v2 as v2

AMENDMENT_1_PATH = r.REPO / "experiments" / "CS_TRANSFER_001" / "AMENDMENT_1.json"
AMENDMENT_2_PATH = r.REPO / "experiments" / "CS_TRANSFER_001" / "AMENDMENT_2.json"

# Rebind the audited runner's data-plane globals to the Amendment-2 surface.
# v2 imports the same `r` module object, so arm execution/load_prepared sees
# these bindings as well.
r.build_shared_surface = data_v2.build_shared_surface
r.deserialize_rows = data_v2.deserialize_rows
r.serialize_rows = data_v2.serialize_rows
r.TokenRow = data_v2.TokenRow
r.COMMON_VOCAB = data_v2.COMMON_VOCAB
r.SELECT_COUNTS = data_v2.SELECT_COUNTS


def _load_amendment(path, number: int) -> dict:
    if not path.is_file():
        raise SystemExit(f"FAIL_CLOSED PROTOCOL: Amendment {number} missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "anra-cs-transfer-001-amendment/v1":
        raise SystemExit(f"FAIL_CLOSED PROTOCOL: Amendment {number} schema")
    if value.get("amendment") != number:
        raise SystemExit(f"FAIL_CLOSED PROTOCOL: Amendment {number} number mismatch")
    if value.get("status") != "PROSPECTIVE_PREEXECUTION":
        raise SystemExit(f"FAIL_CLOSED PROTOCOL: Amendment {number} not prospective")
    if value.get("outcomes_observed_before_amendment") is not False:
        raise SystemExit(f"FAIL_CLOSED PROTOCOL: Amendment {number} outcome provenance")
    return value


def protocol_sha256() -> str:
    _load_amendment(AMENDMENT_1_PATH, 1)
    _load_amendment(AMENDMENT_2_PATH, 2)
    digest = hashlib.sha256()
    digest.update(r.PREREG_PATH.read_bytes())
    digest.update(b"\x00CS_TRANSFER_001_AMENDMENT_1\x00")
    digest.update(AMENDMENT_1_PATH.read_bytes())
    digest.update(b"\x00CS_TRANSFER_001_AMENDMENT_2\x00")
    digest.update(AMENDMENT_2_PATH.read_bytes())
    return digest.hexdigest()


def assert_effective_protocol() -> dict:
    a1 = _load_amendment(AMENDMENT_1_PATH, 1)
    a2 = _load_amendment(AMENDMENT_2_PATH, 2)
    if data_v2.COMMON_VOCAB != 4096:
        raise SystemExit("FAIL_CLOSED PROTOCOL: common vocabulary drift")
    if min(data_v2.GRAMMAR_IDS) < 4 or max(data_v2.GRAMMAR_IDS) >= 4096:
        raise SystemExit("FAIL_CLOSED PROTOCOL: direct-token grammar escaped common space")
    return {
        "schema": "anra-cs-transfer-001-effective-protocol/v2",
        "preregistration_file_sha256": hashlib.sha256(r.PREREG_PATH.read_bytes()).hexdigest(),
        "amendment_1_file_sha256": hashlib.sha256(AMENDMENT_1_PATH.read_bytes()).hexdigest(),
        "amendment_2_file_sha256": hashlib.sha256(AMENDMENT_2_PATH.read_bytes()).hexdigest(),
        "effective_protocol_sha256": protocol_sha256(),
        "amendment_1_status": a1["status"],
        "amendment_2_status": a2["status"],
        "surface_revision": "A2_DIRECT_TOKEN_COMMON_SPACE",
        "direct_token_construction": True,
        "minimum_content_id": min(data_v2.GRAMMAR_IDS),
        "maximum_content_id": max(data_v2.GRAMMAR_IDS),
        "scientific_treatment": "physical tied vocabulary_size 4096 vs 24576 only",
    }


# Bind every persisted identity to preregistration + both prospective amendments.
r._prereg_sha = protocol_sha256


def prepare() -> dict:
    protocol = assert_effective_protocol()
    out = r.prepare_data()
    r._write_receipt("PROTOCOL", protocol)
    return {"mode": "prepare", "protocol": protocol, "data_receipt": out["receipt"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=("prepare", "preflight", "scan", "run-arm", "development", "finalize", "protocol"),
    )
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--arm", choices=r.ARMS, default="PHYS_4096")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    protocol = assert_effective_protocol()
    if args.mode == "protocol":
        result = protocol
    elif args.mode == "prepare":
        result = prepare()
    elif args.mode == "preflight":
        result = r.preflight(cuda=args.cuda)
    elif args.mode == "scan":
        result = r.scan()
    elif args.mode == "run-arm":
        result = v2.run_arm(pair_index=args.pair_index, arm=args.arm, cuda=args.cuda)
    elif args.mode == "development":
        result = r.development_aggregate()
    else:
        result = r.finalize(cuda=args.cuda)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
