import torch

from anra_core.config import CANONICAL_CONFIG
from anra_core.contracts import (
    ArchitectureIdentity,
    CapabilitySet,
    RepresentationIdentity,
    RuntimeIdentity,
)
from anra_core.executor import CoreExecutor
from anra_core.model import AnRaCore
from anra_core.tokenizer import V4Tokenizer


def test_executor_identities_and_contracts(tmp_path) -> None:
    tok = V4Tokenizer.load("anra_core/assets/tokenizer_v4_32k.json")
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model, tokenizer=tok)

    arch_ident = executor.architecture_identity()
    assert isinstance(arch_ident, ArchitectureIdentity)
    assert arch_ident.dense_parameter_count == 180_093_312
    assert arch_ident.vocab_size == 32_768
    assert arch_ident.d_model == 896
    assert arch_ident.n_layers == 18

    rep_ident = executor.representation_identity()
    assert isinstance(rep_ident, RepresentationIdentity)
    assert rep_ident.vocab_size == 32_768
    assert rep_ident.schema_version == 4

    runtime_ident = executor.runtime_identity
    assert isinstance(runtime_ident, RuntimeIdentity)
    assert runtime_ident.runtime_version == "0.5.0"
    assert arch_ident.architecture_sha256 == CANONICAL_CONFIG.architecture_sha256

    caps = executor.capabilities
    assert isinstance(caps, CapabilitySet)
    assert caps.supports_full_forward
    assert caps.supports_incremental_decode
    assert caps.supports_state_fork
    assert caps.supports_state_reset

    desc = executor.describe()
    assert "runtime" in desc
    assert "architecture" in desc
    assert "representation" in desc
    assert "capabilities" in desc


def test_deterministic_execution_profile() -> None:
    torch.manual_seed(999)
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    token_ids = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 12))

    res1 = executor.forward(token_ids)
    res2 = executor.forward(token_ids)

    assert torch.equal(res1.logits, res2.logits)
    assert res1.execution_profile_id == res2.execution_profile_id
