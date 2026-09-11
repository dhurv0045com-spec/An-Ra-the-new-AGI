"""ARK-021 core tests — classification can fail."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "ARK-021"))
import ark021_core as K


class TestArk021(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(K.classify([0.9, 0.92], None), "PRESERVED")
        self.assertEqual(K.classify([0.1, 0.05], 40), "RECONSTRUCTED")
        self.assertEqual(K.classify([0.1, 0.05], None), "RECONSTRUCTED_SLOW_OR_UNSPECIFIED")
        self.assertEqual(K.classify([0.5, 0.6], 20), "MIXED")
        self.assertEqual(K.classify([], None), "INSUFFICIENT_PROBES")

    def test_probe_steps(self):
        self.assertEqual(K.probe_steps(600, 200), [200, 400, 600])
        self.assertEqual(K.probe_steps(500, 200), [200, 400])

    def test_classification_can_fail(self):
        # mutation guard: a deliberately wrong threshold flips the label
        self.assertNotEqual(K.classify([0.9, 0.9], None),
                            K.classify([0.2, 0.2], None))


if __name__ == "__main__":
    unittest.main()
