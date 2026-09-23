"""B01 focused tests: strict configuration, parameter accounting, seeding."""
import unittest

from bramastra_lab.research.config import (
    BYTE_TOKENIZER_NAME,
    BuildConfig,
    ConfigError,
    FeatureSection,
    ModelSection,
    seed_everything,
    tokenizer_identity,
)

TINY = {"profile": "tiny", "vocab": 260, "layers": 2, "width": 64, "heads": 4,
        "ffn": 176, "max_seq": 128}


def tiny_config() -> BuildConfig:
    return BuildConfig.from_dict({"model": {"profile": "tiny"}})


class ModelSectionTests(unittest.TestCase):
    def test_profiles_resolve_to_contract_numbers(self) -> None:
        self.assertEqual(ModelSection.from_dict({"profile": "tiny"}).__dict__, {
            "profile": "tiny", "vocab": 260, "layers": 2, "width": 64, "heads": 4,
            "ffn": 176, "max_seq": 128, "tokenizer": BYTE_TOKENIZER_NAME,
        })
        development = ModelSection.from_dict({"profile": "development"})
        self.assertEqual((development.layers, development.width, development.heads,
                          development.ffn, development.max_seq), (8, 256, 4, 704, 256))
        capacity = ModelSection.from_dict({"profile": "future_capacity"})
        self.assertEqual((capacity.layers, capacity.width, capacity.heads,
                          capacity.ffn, capacity.max_seq), (12, 512, 8, 1408, 512))

    def test_unknown_profile_and_unknown_keys_reject(self) -> None:
        with self.assertRaises(ConfigError):
            ModelSection.from_dict({"profile": "giant"})
        with self.assertRaises(ConfigError):
            ModelSection.from_dict({"profile": "tiny", "surprise": 1})
        with self.assertRaises(ConfigError):
            BuildConfig.from_dict({"model": {"profile": "tiny"}, "extra_section": {}})

    def test_invalid_head_geometry_rejects(self) -> None:
        with self.assertRaises(ConfigError):
            ModelSection.from_dict({"profile": "tiny", "width": 66, "heads": 4})
        with self.assertRaises(ConfigError):
            ModelSection.from_dict({"profile": "tiny", "width": 44, "heads": 4})  # odd per-head

    def test_sub_contract_vocabulary_rejects(self) -> None:
        with self.assertRaises(ConfigError):
            ModelSection.from_dict({"profile": "tiny", "vocab": 128})

    def test_unknown_tokenizer_rejects(self) -> None:
        with self.assertRaises(ConfigError):
            ModelSection.from_dict({"profile": "tiny", "tokenizer": "bpe-4096/v1"})


class BuildConfigTests(unittest.TestCase):
    def test_defaults_match_build_contract(self) -> None:
        config = tiny_config()
        self.assertEqual(config.training.logit_treatment, "full")
        self.assertEqual(config.training.pair_loss_weight, 0.0)
        self.assertEqual(config.controller.mode, "disabled")
        self.assertFalse(config.features.retrieval_enabled)
        self.assertEqual(config.features.planner, "none")
        self.assertEqual(config.features.primary_generation, "full_vocabulary")
        self.assertTrue(config.features.require_eos)
        self.assertFalse(config.features.automatic_training_launch)

    def test_automatic_training_launch_rejects(self) -> None:
        with self.assertRaises(ConfigError):
            BuildConfig.from_dict({"model": {"profile": "tiny"},
                                   "features": {"automatic_training_launch": True}})

    def test_identity_is_stable_and_distinguishes_changes(self) -> None:
        base = tiny_config()
        self.assertEqual(base.identity(), tiny_config().identity())
        wider = BuildConfig.from_dict({"model": {"profile": "tiny", "width": 128}})
        more_heads = BuildConfig.from_dict({"model": {"profile": "tiny", "heads": 8,
                                                      "width": 64}})  # 16 per head, even
        self.assertNotEqual(base.identity(), wider.identity())
        self.assertNotEqual(base.identity(), more_heads.identity())
        paired = BuildConfig.from_dict({"model": {"profile": "tiny"},
                                        "training": {"pair_loss_weight": 0.5}})
        self.assertNotEqual(base.identity(), paired.identity())

    def test_pair_loss_and_controller_validation(self) -> None:
        with self.assertRaises(ConfigError):
            BuildConfig.from_dict({"model": {"profile": "tiny"},
                                   "training": {"pair_loss_weight": -0.1}})
        with self.assertRaises(ConfigError):
            BuildConfig.from_dict({"model": {"profile": "tiny"},
                                   "training": {"logit_treatment": "banana"}})
        with self.assertRaises(ConfigError):
            BuildConfig.from_dict({"model": {"profile": "tiny"},
                                   "training": {"logit_treatment": "inactive_offset"}})
        with self.assertRaises(ConfigError):
            BuildConfig.from_dict({"model": {"profile": "tiny"},
                                   "training": {"logit_treatment": "inactive_offset",
                                                "effective_vocab": 1024}})  # > vocab 260
        valid = BuildConfig.from_dict({"model": {"profile": "tiny"},
                                       "training": {"logit_treatment": "inactive_offset",
                                                    "effective_vocab": 64}})
        self.assertEqual(valid.training.effective_vocab, 64)

    def test_controller_requires_pool_and_threshold_ordering(self) -> None:
        with self.assertRaises(ConfigError):
            BuildConfig.from_dict({"model": {"profile": "tiny"},
                                   "controller": {"mode": "evidence_driven"}})
        with self.assertRaises(ConfigError):
            BuildConfig.from_dict({
                "model": {"profile": "tiny"},
                "controller": {"mode": "evidence_driven", "controller_pool_id": "pool-a",
                               "formation_threshold": 0.5, "reacquire_threshold": 0.7}})
        ok = BuildConfig.from_dict({
            "model": {"profile": "tiny"},
            "controller": {"mode": "evidence_driven", "controller_pool_id": "pool-a"}})
        self.assertEqual(ok.controller.controller_pool_id, "pool-a")

    def test_future_capacity_is_configuration_only(self) -> None:
        config = BuildConfig.from_dict({"model": {"profile": "future_capacity"}})
        # Analytic validation must not require loading the model on a small device.
        self.assertEqual(config.base_decoder_parameter_count(), 38_681_088)
        self.assertEqual(config.parameter_count(), 38_681_088 + 2 * 512)

    def test_tpu_100m_profile_is_analytic_and_exact(self) -> None:
        config = BuildConfig.from_dict({"model": {"profile": "tpu_100m"}})
        self.assertEqual(
            (config.model.layers, config.model.width, config.model.heads,
             config.model.ffn, config.model.max_seq),
            (15, 640, 10, 2624, 512))
        self.assertEqual(config.parameter_count(), 100_334_720)
        self.assertEqual(config.base_decoder_parameter_count(), 100_333_440)


class ParameterAccountingTests(unittest.TestCase):
    """Analytic counts derived independently of the decoder implementation."""

    def test_declared_profile_counts(self) -> None:
        self.assertEqual(tiny_config().base_decoder_parameter_count(), 117_312)
        self.assertEqual(
            BuildConfig.from_dict({"model": {"profile": "development"}})
            .base_decoder_parameter_count(), 6_493_440)

    def test_integrated_count_adds_head_parameters_exactly(self) -> None:
        config = tiny_config()
        # base: 260*64 + 2*(4*64^2 + 3*64*176 + 2*64) + 64 = 117312 (tied output)
        base = 260 * 64 + 2 * (4 * 64 * 64 + 3 * 64 * 176 + 2 * 64) + 64
        self.assertEqual(config.base_decoder_parameter_count(), base)
        self.assertEqual(config.parameter_count(), base + 2 * 64)

    def test_translated_model_config_matches_section(self) -> None:
        from bramastra_lab.model import ModelConfig

        config = tiny_config()
        translated = config.model_config()
        self.assertIsInstance(translated, ModelConfig)
        self.assertEqual((translated.vocab, translated.width, translated.layers,
                          translated.heads, translated.ffn, translated.max_seq),
                         (260, 64, 2, 4, 176, 128))


class TokenizerIdentityTests(unittest.TestCase):
    def test_tokenizer_identity_is_stable_string(self) -> None:
        self.assertEqual(tokenizer_identity(), tokenizer_identity())
        self.assertIsInstance(tokenizer_identity(), str)


class SeedingTests(unittest.TestCase):
    def test_repeated_seed_initialization_matches(self) -> None:
        import random

        import torch

        seed_everything(1234)
        python_first = [random.random() for _ in range(4)]
        torch_first = torch.rand(4)
        seed_everything(1234)
        python_second = [random.random() for _ in range(4)]
        torch_second = torch.rand(4)
        self.assertEqual(python_first, python_second)
        self.assertTrue(torch.equal(torch_first, torch_second))
        seed_everything(4321)
        self.assertNotEqual([random.random() for _ in range(4)], python_first)

    def test_seed_rejects_invalid(self) -> None:
        with self.assertRaises(ConfigError):
            seed_everything(-1)


if __name__ == "__main__":
    unittest.main()
