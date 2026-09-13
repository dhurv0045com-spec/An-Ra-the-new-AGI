"""Canonical pre-execution CS-TRANSFER-001 runner.

Revision v3 binds prospective Amendment 1 into every persistent experiment
identity.  Amendment 1 changed only the answer boundary used by the low-ID
data instrument after a static tokenizer audit proved the original literal
``Answer:`` delimiter contains production-tokenizer ID 18224 and therefore
cannot satisfy the frozen <4096 shared-token requirement.  No model outcome
was observed before the amendment.

Scientific arm execution itself is the audited v2 implementation.  All
persistent prepare/resume/finalize modes in this module use a composite
protocol SHA over the original preregistration plus Amendment 1, so a Drive
campaign created under the pre-amendment protocol cannot be resumed silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from anra_v5 import cs_transfer_001_data as data
from anra_v5 import cs_transfer_001_run as r
from anra_v5 import cs_transfer_001_run_v2 as v2

AMENDMENT_PATH = r.REPO / "experiments" / "CS_TRANSFER_001" / "AMENDMENT_1.json"


def protocol_sha256() -> str:
    if not AMENDMENT_PATH.is_file():
        raise SystemExit("FAIL_CLOSED PROTOCOL: Amendment 1 is missing")
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if amendment.get("schema") != "anra-cs-transfer-001-amendment/v1":
        raise SystemExit("FAIL_CLOSED PROTOCOL: wrong Amendment 1 schema")
    if amendment.get("status") != "PROSPECTIVE_PREEXECUTION":
        raise SystemExit("FAIL_CLOSED PROTOCOL: Amendment 1 is not prospective")
    if amendment.get("outcomes_observed_before_amendment") is not False:
        raise SystemExit("FAIL_CLOSED PROTOCOL: amendment provenance is not pre-outcome")
    digest = hashlib.sha256()
    digest.update(r.PREREG_PATH.read_bytes())
    digest.update(b"\x00CS_TRANSFER_001_AMENDMENT_1\x00")
    digest.update(AMENDMENT_PATH.read_bytes())
    return digest.hexdigest()


def assert_effective_protocol() -> dict:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    change = amendment["change"]
    if data.PROMPT_SUFFIX != change["prompt_suffix_after"]:
        raise SystemExit("FAIL_CLOSED PROTOCOL: data prompt suffix disagrees with Amendment 1")
    if data.ANSWER_PREFIX != change["answer_prefix_after"]:
        raise SystemExit("FAIL_CLOSED PROTOCOL: data answer prefix disagrees with Amendment 1")
    if data.PROMPT_SUFFIX != "\n" or data.ANSWER_PREFIX != "":
        raise SystemExit("FAIL_CLOSED PROTOCOL: unexpected effective answer boundary")
    return {
        "schema": "anra-cs-transfer-001-effective-protocol/v1",
        "preregistration_file_sha256": hashlib.sha256(r.PREREG_PATH.read_bytes()).hexdigest(),
        "amendment_1_file_sha256": hashlib.sha256(AMENDMENT_PATH.read_bytes()).hexdigest(),
        "effective_protocol_sha256": protocol_sha256(),
        "prompt_suffix": data.PROMPT_SUFFIX,
        "answer_prefix": data.ANSWER_PREFIX,
        "amendment_status": amendment["status"],
    }


# The base implementation consistently calls r._prereg_sha() for data receipt,
# prepared-state compatibility, run-spec identity, and final result identity.
# Rebinding it here makes every v3 entry point amendment-aware without changing
# the historical unexecuted v1/v2 source files.
r._prereg_sha = protocol_sha256


def prepare() -> dict:
    protocol = assert_effective_protocol()
    out = r.prepare_data()
    # Separate immutable protocol receipt makes the composite identity legible
    # to operators instead of hiding it behind a legacy field name.
    r._write_receipt("PROTOCOL", protocol)
    return {
        "mode": "prepare",
        "protocol": protocol,
        "data_receipt": out["receipt"],
    }


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
