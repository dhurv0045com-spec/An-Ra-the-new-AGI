"""Pure, checkpoint-driven sampler planning for production updates.

The planner is the single authority for converting a TrainingState plus its
bucket cursor into the exact next logical update. It stages all cursor and
mixture changes locally, so validation or data-supply errors cannot mutate the
live campaign state. The optimizer loop and resume verifier consume the same
windows and shared rank materializer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from v5_data.bucket_cursor import (
    BUCKET_CURSOR_SCHEMA,
    BucketCursorState,
    LaneWindow,
    build_bucket_lanes,
    cell_key,
    lane_remainder,
    take_cell_window,
)
from v5_data.mixture import DeficitScheduler
from v5_training.production_microsteps import (
    MaterializedMicrostep,
    count_eligible_targets,
    materialize_rank_microsteps,
)
from v5_training.state import TrainingState, next_update_tokens


PRODUCTION_SAMPLER_SCHEMA = "anra-v5-production-sampler/v1"
VERIFIED_COGNITION = "verified_cognition"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def campaign_microstep_plan(*, start_tokens: int, campaign_tokens: int,
                            topo: Mapping[str, Any]) -> list[tuple[int, int]]:
    """Exact campaign-remainder (bucket, real-token-count) sequence."""

    per_update = topo["global_tokens_per_update"]
    microstep = topo["global_tokens_per_microstep"]
    cycle = topo["supercycle"]
    if type(start_tokens) is not int or type(campaign_tokens) is not int:
        raise ValueError("campaign token positions must be integers")
    if not 0 <= start_tokens <= campaign_tokens:
        raise ValueError("campaign remainder is outside the budget")
    if type(per_update) is not int or per_update <= 0:
        raise ValueError("global tokens per update must be positive")
    if type(microstep) is not int or microstep <= 0:
        raise ValueError("global tokens per microstep must be positive")
    if not isinstance(cycle, (tuple, list)) or not cycle or any(
        type(bucket) is not int or bucket <= 0 for bucket in cycle
    ):
        raise ValueError("frozen bucket supercycle must be nonempty and positive")

    plan: list[tuple[int, int]] = []
    ordinal = start_tokens // microstep
    remaining = campaign_tokens - start_tokens
    while remaining > 0:
        expected = min(per_update, remaining)
        full, tail = divmod(expected, microstep)
        counts = [microstep] * full + ([tail] if tail else [])
        for count in counts:
            plan.append((int(cycle[ordinal % len(cycle)]), count))
            ordinal += 1
        remaining -= expected
    return plan


def partial_microstep_plan(
    *, remaining_tokens: int, topo: Mapping[str, Any], microstep_ordinal: int,
) -> list[int]:
    """Split one logical update into full global microsteps and an exact tail."""

    if type(microstep_ordinal) is not int or microstep_ordinal < 0:
        raise ValueError("microstep ordinal must be a nonnegative integer")
    microstep = topo["global_tokens_per_microstep"]
    if type(microstep) is not int or microstep <= 0:
        raise ValueError("global tokens per microstep must be positive")
    if type(remaining_tokens) is not int or remaining_tokens <= 0:
        raise ValueError("remaining tokens must be positive")
    full, tail = divmod(remaining_tokens, microstep)
    return [microstep] * full + ([tail] if tail else [])


def microstep_buckets(
    *, cumulative_tokens: int, microstep_counts: list[int], topo: Mapping[str, Any],
) -> list[int]:
    """Derive each bucket from the cumulative-token supercycle position."""

    if type(cumulative_tokens) is not int or cumulative_tokens < 0:
        raise ValueError("cumulative tokens must be a nonnegative integer")
    cycle = topo["supercycle"]
    if not cycle:
        raise ValueError("frozen bucket supercycle cannot be empty")
    base = cumulative_tokens // topo["global_tokens_per_microstep"]
    return [int(cycle[(base + index) % len(cycle)])
            for index in range(len(microstep_counts))]


def assign_mixture_cell(*, fam_consumed: dict[str, int],
                        sub_consumed: dict[str, int],
                        total_consumed: int,
                        fam_scheduler: DeficitScheduler | None,
                        sub_scheduler: DeficitScheduler | None,
                        cognition_mapped: bool) -> tuple[str, str]:
    """Pure per-microstep family/subfamily choice from checkpoint counters."""

    if fam_scheduler is None:
        return "", ""
    family = fam_scheduler.next(consumed_total=total_consumed,
                                consumed=fam_consumed)
    sub = ""
    if family == VERIFIED_COGNITION and cognition_mapped and sub_scheduler is not None:
        sub = sub_scheduler.next(
            consumed_total=fam_consumed.get(family, 0), consumed=sub_consumed)
    return family, sub


def plan_campaign_demand(*, microstep_plan: list[tuple[int, int]],
                         fam_scheduler: DeficitScheduler | None,
                         sub_scheduler: DeficitScheduler | None,
                         cognition_mapped: bool,
                         initial_fam_consumed: dict[str, int] | None = None,
                         initial_sub_consumed: dict[str, int] | None = None,
                         initial_total: int = 0) -> dict[str, int]:
    """Pure exact per-cell supply demand used by startup preflight."""

    fam_consumed = dict(initial_fam_consumed or {})
    sub_consumed = dict(initial_sub_consumed or {})
    total = initial_total
    if type(total) is not int or total < 0:
        raise ValueError("planning total must be a nonnegative integer")
    demand: dict[str, int] = {}
    for bucket, count in microstep_plan:
        family, sub = assign_mixture_cell(
            fam_consumed=fam_consumed, sub_consumed=sub_consumed,
            total_consumed=total, fam_scheduler=fam_scheduler,
            sub_scheduler=sub_scheduler, cognition_mapped=cognition_mapped)
        key = cell_key(bucket, family, sub)
        demand[key] = demand.get(key, 0) + count
        fam_consumed[family] = fam_consumed.get(family, 0) + count
        if sub:
            sub_consumed[sub] = sub_consumed.get(sub, 0) + count
        total += count
    return dict(sorted(demand.items()))


@dataclass(frozen=True, slots=True)
class PlannedProductionUpdate:
    expected_real_tokens: int
    windows: tuple[tuple[int, str, str, LaneWindow], ...]
    rank_microsteps: tuple[MaterializedMicrostep, ...]
    eligible_tokens: int
    end_cursor: BucketCursorState
    replay_events: tuple[Mapping[str, int], ...]
    lanes_receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ProductionSampler:
    """Immutable runtime data needed to deterministically rebuild the sampler."""

    packed: Sequence[Any]
    run_seed: int
    topology: Mapping[str, Any]
    pack_manifest_sha256: str
    cell_of_source: Mapping[str, tuple[str, str]] | None
    mixture_fractions: Mapping[str, float] | None
    cognition_fractions: Mapping[str, float] | None
    cognition_mapped: bool
    allow_replay: bool

    def __post_init__(self) -> None:
        if type(self.run_seed) is not int or self.run_seed < 0:
            raise ValueError("sampler run seed must be a nonnegative integer")
        if type(self.allow_replay) is not bool or type(self.cognition_mapped) is not bool:
            raise ValueError("sampler policy flags must be booleans")
        if len(self.pack_manifest_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.pack_manifest_sha256
        ):
            raise ValueError("sampler pack identity must be lowercase SHA-256")
        if not self.packed:
            raise ValueError("production sampler requires a nonempty immutable pack")
        topo = dict(self.topology)
        topo["supercycle"] = tuple(topo["supercycle"])
        if "sequences_per_replica_by_bucket" in topo:
            topo["sequences_per_replica_by_bucket"] = MappingProxyType(
                dict(topo["sequences_per_replica_by_bucket"])
            )
        object.__setattr__(self, "packed", tuple(self.packed))
        object.__setattr__(self, "topology", MappingProxyType(topo))
        object.__setattr__(self, "cell_of_source", (
            MappingProxyType({key: tuple(value) for key, value in self.cell_of_source.items()})
            if self.cell_of_source is not None else None
        ))
        object.__setattr__(self, "mixture_fractions", (
            MappingProxyType(dict(self.mixture_fractions))
            if self.mixture_fractions is not None else None
        ))
        object.__setattr__(self, "cognition_fractions", (
            MappingProxyType(dict(self.cognition_fractions))
            if self.cognition_fractions is not None else None
        ))
        if type(topo.get("replicas")) is not int or topo["replicas"] <= 0:
            raise ValueError("sampler topology needs a positive replica count")

    @property
    def sha256(self) -> str:
        """Identity for policies that change the examples or their ordering."""

        cell_map = ({key: list(value) for key, value in sorted(self.cell_of_source.items())}
                    if self.cell_of_source is not None else None)
        topology = {
            "replicas": self.topology["replicas"],
            "global_tokens_per_microstep": self.topology["global_tokens_per_microstep"],
            "global_tokens_per_update": self.topology["global_tokens_per_update"],
            "supercycle": list(self.topology["supercycle"]),
        }
        return _sha256(_canonical_json({
            "schema": PRODUCTION_SAMPLER_SCHEMA,
            "pack_manifest_sha256": self.pack_manifest_sha256,
            "run_seed": self.run_seed,
            "topology": topology,
            "cell_map": cell_map,
            "mixture_fractions": (dict(self.mixture_fractions)
                                  if self.mixture_fractions is not None else None),
            "cognition_fractions": (dict(self.cognition_fractions)
                                    if self.cognition_fractions is not None else None),
            "cognition_mapped": self.cognition_mapped,
            "allow_replay": self.allow_replay,
            "pad_id": 0,
        }))

    def materialize_update(
        self, state: TrainingState, *, rank: int | None = None,
    ) -> PlannedProductionUpdate:
        """Recreate the exact next bucket windows without changing ``state``."""

        state.assert_valid()
        if (state.identities.sampler_spec_sha256 is not None
                and state.identities.sampler_spec_sha256 != self.sha256):
            raise ValueError("production sampler identity differs from training state")
        cursor = state.cursor
        if not isinstance(cursor, BucketCursorState):
            raise ValueError("production sampler requires a bucket-lane cursor")
        cursor.assert_valid()
        if state.identities.pack_manifest_sha256 != self.pack_manifest_sha256:
            raise ValueError("sampler pack identity differs from training state")
        if cursor.pack_manifest_sha256 != self.pack_manifest_sha256:
            raise ValueError("sampler pack identity differs from bucket cursor")
        if state.tokens_per_update != self.topology["global_tokens_per_update"]:
            raise ValueError("sampler update size differs from training state")
        replicas = self.topology["replicas"]
        if rank is not None and (type(rank) is not int or not 0 <= rank < replicas):
            raise ValueError("sampler rank is outside the configured replica world")

        campaign_plan = campaign_microstep_plan(
            start_tokens=state.cumulative_tokens,
            campaign_tokens=state.token_budget,
            topo=self.topology,
        )
        if not campaign_plan:
            raise ValueError("complete training state has no next sampler update")
        required_buckets = {bucket for bucket, _count in campaign_plan}
        lanes, receipt = build_bucket_lanes(
            self.packed,
            run_seed=self.run_seed,
            pattern=list(self.topology["supercycle"]),
            epoch=cursor.epoch,
            cell_of_source=self.cell_of_source,
            required_buckets=required_buckets,
        )
        if receipt["lanes_sha256"] != cursor.lanes_sha256:
            raise ValueError("rebuilt production lanes disagree with checkpoint cursor")
        if set(cursor.positions) != set(lanes):
            raise ValueError("checkpoint cursor positions do not cover the rebuilt lane set")
        for key, position in cursor.positions.items():
            lane_index, token_offset = int(position[0]), int(position[1])
            lane = lanes[key]
            if lane_index > len(lane) or (lane_index == len(lane) and token_offset != 0):
                raise ValueError(f"checkpoint cursor has an invalid coordinate for lane {key}")
            lane_remainder(self.packed, lane, lane_index, token_offset, pad=0)

        positions = {key: [int(value[0]), int(value[1])]
                     for key, value in cursor.positions.items()}
        fam_consumed = dict(cursor.mixture_consumed)
        sub_consumed = dict(cursor.sub_consumed)
        epoch = cursor.epoch
        current_receipt = receipt
        replay_events: list[dict[str, int]] = []
        staged_total = state.cumulative_tokens
        family_scheduler = (
            DeficitScheduler(
                fractions=dict(self.mixture_fractions),
                order=tuple(sorted(self.mixture_fractions)),
            ) if self.mixture_fractions is not None else None
        )
        subfamily_scheduler = (
            DeficitScheduler(fractions=dict(self.cognition_fractions))
            if self.cognition_fractions is not None else None
        )

        def take_window(bucket: int, family: str, subfamily: str, count: int) -> LaneWindow:
            nonlocal lanes, current_receipt, epoch
            key = cell_key(bucket, family, subfamily)
            if key not in lanes:
                raise ValueError(
                    f"abort DATA_NOT_READY: no packed supply for cell {key}; "
                    "no silent substitution across buckets or families")
            if not lanes[key]:
                raise ValueError(f"abort DATA_NOT_READY: lane {key} holds no sequences")
            rows_tokens: list[tuple[int, ...]] = []
            rows_segments: list[tuple[int, ...]] = []
            rows_eligible: list[tuple[bool, ...]] = []
            row_widths: list[int] = []
            merged_source: dict[str, int] = {}
            merged_family: dict[str, int] = {}
            merged_real = 0
            end_index, end_offset = positions[key]
            remaining = count
            for _attempt in range(1024):
                lane = lanes[key]
                available = lane_remainder(
                    self.packed, lane, end_index, end_offset, pad=0,
                )
                if available <= 0:
                    if not self.allow_replay:
                        raise ValueError(
                            f"abort DATA_NOT_READY: lane exhausted for cell {key}; "
                            "bounded replay not permitted")
                    epoch += 1
                    lanes, current_receipt = build_bucket_lanes(
                        self.packed,
                        run_seed=self.run_seed,
                        pattern=list(self.topology["supercycle"]),
                        epoch=epoch,
                        cell_of_source=self.cell_of_source,
                        required_buckets=required_buckets,
                    )
                    for reset_key in positions:
                        positions[reset_key] = [0, 0]
                    replay_events.append({
                        "epoch": epoch,
                        "at_cumulative_tokens": staged_total,
                    })
                    lane = lanes[key]
                    if not lane:
                        raise ValueError(
                            f"abort DATA_NOT_READY: cell {key} cannot supply "
                            f"{count} real tokens even fresh")
                    end_index, end_offset = 0, 0
                    continue
                fragment = take_cell_window(
                    self.packed, lane, end_index, end_offset,
                    real_tokens=min(remaining, available), pad=0, bucket=bucket,
                    cell_of_source=self.cell_of_source)
                rows_tokens.extend(fragment.tokens)
                rows_segments.extend(fragment.segment_ids)
                rows_eligible.extend(fragment.eligible)
                row_widths.extend(fragment.row_widths)
                for source, amount in fragment.tokens_by_source.items():
                    merged_source[source] = merged_source.get(source, 0) + amount
                for name, amount in fragment.tokens_by_family.items():
                    merged_family[name] = merged_family.get(name, 0) + amount
                merged_real += fragment.real_tokens
                remaining -= fragment.real_tokens
                end_index, end_offset = fragment.end_lane_index, fragment.end_token_offset
                if remaining <= 0:
                    break
            if remaining > 0:
                raise ValueError(
                    f"abort DATA_NOT_READY: cell {key} cannot supply {count} real tokens")
            if merged_real != count:
                raise ValueError("sampler window did not fulfill its exact real-token budget")
            positions[key] = [end_index, end_offset]
            return LaneWindow(
                tokens=tuple(rows_tokens),
                segment_ids=tuple(rows_segments),
                eligible=tuple(rows_eligible),
                tokens_by_source=dict(sorted(merged_source.items())),
                tokens_by_family=dict(sorted(merged_family.items())),
                real_tokens=merged_real,
                row_widths=tuple(row_widths),
                end_lane_index=end_index,
                end_token_offset=end_offset,
            )

        expected = next_update_tokens(
            token_budget=state.token_budget,
            cumulative_tokens=state.cumulative_tokens,
            tokens_per_update=state.tokens_per_update,
        )
        counts = partial_microstep_plan(
            remaining_tokens=expected,
            topo=self.topology,
            microstep_ordinal=(state.cumulative_tokens
                               // self.topology["global_tokens_per_microstep"]),
        )
        buckets = microstep_buckets(
            cumulative_tokens=state.cumulative_tokens,
            microstep_counts=counts,
            topo=self.topology,
        )
        if len(counts) != len(buckets):
            raise ValueError("microstep plan and bucket plan lengths differ")

        windows: list[tuple[int, str, str, LaneWindow]] = []
        for micro_count, bucket in zip(counts, buckets):
            family, subfamily = assign_mixture_cell(
                fam_consumed=fam_consumed,
                sub_consumed=sub_consumed,
                total_consumed=staged_total,
                fam_scheduler=family_scheduler,
                sub_scheduler=subfamily_scheduler,
                cognition_mapped=self.cognition_mapped,
            )
            window = take_window(bucket, family, subfamily, micro_count)
            windows.append((bucket, family, subfamily, window))
            fam_consumed[family] = fam_consumed.get(family, 0) + window.real_tokens
            if subfamily:
                sub_consumed[subfamily] = sub_consumed.get(subfamily, 0) + window.real_tokens
            staged_total += window.real_tokens

        actual_tokens = sum(window.real_tokens for _, _, _, window in windows)
        if actual_tokens != expected:
            raise ValueError("sampler microsteps disagree with the logical update budget")
        eligible_tokens = sum(count_eligible_targets(window)
                              for _, _, _, window in windows)
        if eligible_tokens <= 0:
            raise ValueError("abort NO_SUPERVISED_TOKENS: update carried no eligible targets")

        end_cursor = BucketCursorState(
            schema=BUCKET_CURSOR_SCHEMA,
            pack_manifest_sha256=self.pack_manifest_sha256,
            lanes_sha256=str(current_receipt["lanes_sha256"]),
            positions=positions,
            mixture_consumed=fam_consumed,
            sub_consumed=sub_consumed,
            epoch=epoch,
            replay_count=cursor.replay_count + len(replay_events),
        )
        end_cursor.assert_valid()
        rank_microsteps = materialize_rank_microsteps(
            windows,
            replicas=replicas,
            rank=rank,
            planned_total=eligible_tokens,
        )
        return PlannedProductionUpdate(
            expected_real_tokens=expected,
            windows=tuple(windows),
            rank_microsteps=rank_microsteps,
            eligible_tokens=eligible_tokens,
            end_cursor=end_cursor,
            replay_events=tuple(MappingProxyType(dict(event)) for event in replay_events),
            lanes_receipt=MappingProxyType(dict(current_receipt)),
        )


__all__ = [
    "PRODUCTION_SAMPLER_SCHEMA",
    "ProductionSampler",
    "PlannedProductionUpdate",
    "assign_mixture_cell",
    "campaign_microstep_plan",
    "microstep_buckets",
    "partial_microstep_plan",
    "plan_campaign_demand",
]
