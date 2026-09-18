"""Frozen HORM-001 experimental spec (sibling of V5A_250M; never a mutation)."""

from __future__ import annotations

from dataclasses import dataclass

from v5_contracts.model_spec import ModelSpec, V5A_250M


@dataclass(frozen=True, slots=True)
class HormonalOverlay:
    """Frozen overlay describing the sibling spec variant."""

    schema: str
    base_spec_sha256: str
    scale_bounds: tuple[float, float]
    bound: float
    raw_alpha: float

    def assert_sibling(self) -> None:
        if self.base_spec_sha256 != V5A_250M.sha256():
            raise ValueError("overlay must bind the frozen V5A_250M base spec hash")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "base_spec_sha256": self.base_spec_sha256,
            "scale_bounds": list(self.scale_bounds),
            "bound": self.bound,
            "raw_alpha": self.raw_alpha,
        }


V5A_250M_HORMONAL_V1 = HormonalOverlay(
    schema="anra-v5-hormonal-overlay/v1",
    base_spec_sha256=V5A_250M.sha256(),
    scale_bounds=(0.8, 1.2),
    bound=0.2,
    raw_alpha=0.0,
)
