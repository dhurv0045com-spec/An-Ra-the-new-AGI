"""Experimental composite spec; not a launch-gated ModelSpec."""
from dataclasses import asdict, dataclass
import hashlib
import json

from v5_contracts.model_spec import ModelSpec, V5A_250M
from v5_identity.hormonal_config import HormonalConfig


@dataclass(frozen=True, slots=True)
class HormonalModelSpec:
    base: ModelSpec = V5A_250M
    hormonal: HormonalConfig = HormonalConfig()
    family: str = "dense-decoder-transformer-hormonal-v1"

    def canonical(self):
        self.base.assert_valid()
        return asdict(self)

    def sha256(self):
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def parameter_receipt(self):
        base = self.base.parameter_receipt().total
        projection = self.base.query_heads * 7
        return {"base": base, "projection": projection, "total": base + projection}


V5A_250M_HORMONAL_V1 = HormonalModelSpec()
V5A_250M_HORMONAL_v1 = V5A_250M_HORMONAL_V1
