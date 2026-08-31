from __future__ import annotations

import json
import unittest
from dataclasses import replace

from v5_training.distributed import DistributedCheckpoint, RankCheckpoint


def _rank(rank: int, *, barrier: str = "c" * 64) -> RankCheckpoint:
    return RankCheckpoint(
        schema="anra-v5-rank-checkpoint/v1",
        rank=rank,
        world_size=2,
        global_update=7,
        token_contribution=8,
        cursor_sha256=f"{rank + 1:064x}",
        rng_state_sha256=f"{rank + 3:064x}",
        optimizer_shard_sha256=f"{rank + 5:064x}",
        data_shard_identity=f"shard-{rank}",
        collective_barrier_sha256=barrier,
    )


class V5DistributedCheckpointTests(unittest.TestCase):
    def test_complete_rank_set_reconciles_tokens_and_round_trips(self) -> None:
        checkpoint = DistributedCheckpoint(
            schema="anra-v5-distributed-checkpoint/v1",
            parent_checkpoint_sha256="a" * 64,
            global_update=7,
            global_tokens=16,
            world_size=2,
            topology="2x-v5-target",
            ranks=(_rank(1), _rank(0)),
        )
        checkpoint.assert_valid()
        encoded = json.loads(json.dumps(checkpoint.canonical(), sort_keys=True))
        self.assertEqual(DistributedCheckpoint.from_dict(encoded), checkpoint)
        self.assertEqual(len(checkpoint.sha256()), 64)

    def test_missing_duplicate_or_misaligned_rank_fails_closed(self) -> None:
        base = DistributedCheckpoint(
            schema="anra-v5-distributed-checkpoint/v1",
            parent_checkpoint_sha256=None,
            global_update=7,
            global_tokens=16,
            world_size=2,
            topology="target",
            ranks=(_rank(0), _rank(1)),
        )
        for ranks in (( _rank(0),), (_rank(0), _rank(0)), (_rank(0), replace(_rank(1), token_contribution=7))):
            with self.assertRaises(ValueError):
                replace(base, ranks=ranks).assert_valid()
        with self.assertRaises(ValueError):
            replace(base, ranks=(_rank(0), replace(_rank(1), collective_barrier_sha256="d" * 64))).assert_valid()
        with self.assertRaises(ValueError):
            replace(base, ranks=(_rank(0), replace(_rank(1), optimizer_shard_sha256=_rank(0).optimizer_shard_sha256))).assert_valid()

    def test_rank_world_and_hashes_are_bound(self) -> None:
        with self.assertRaises(ValueError):
            replace(_rank(0), rank=2).assert_valid()
        with self.assertRaises(ValueError):
            replace(_rank(0), rng_state_sha256="bad").assert_valid()


if __name__ == "__main__":
    unittest.main()
