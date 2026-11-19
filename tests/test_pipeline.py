"""Unit tests for landmark normalization, post-processing, and Deep MLP architecture."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.config import FEATURE_DIM, NUM_CLASSES
from src.features import normalize_landmarks
from src.postprocess import WordFormationProcessor
from src.train_mlp import FSLDeepMLP


class TestFeatureNormalization(unittest.TestCase):
    def test_translation_and_scale_invariance(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)

        translated = base + np.array([2.5, -1.25, 0.4], dtype=np.float32)
        scaled = base * 3.7

        v_base = normalize_landmarks(base)
        v_trans = normalize_landmarks(translated)
        v_scale = normalize_landmarks(scaled)

        self.assertEqual(v_base.shape, (FEATURE_DIM,))
        np.testing.assert_allclose(v_base, v_trans, atol=1e-5)
        np.testing.assert_allclose(v_base, v_scale, atol=1e-5)
        # Wrist (landmark 0) must be at origin
        np.testing.assert_allclose(v_base[:3], np.zeros(3), atol=1e-6)


class TestWordFormationProcessor(unittest.TestCase):
    def test_threshold_smoothing_and_dwell_confirmation(self) -> None:
        proc = WordFormationProcessor(
            confidence_threshold=0.75,
            smoothing_window=10,
            dwell_time_seconds=1.0,
        )

        # Low confidence should not confirm
        for i in range(15):
            st = proc.update("A", 0.50, timestamp_sec=i * 0.1)
            self.assertIsNone(st.just_confirmed)
        self.assertEqual(proc.get_text(), "")

        # High confidence held for >= 1.0s should confirm 'A' once
        confirmed = []
        for i in range(20):
            st = proc.update("A", 0.92, timestamp_sec=2.0 + i * 0.1)
            if st.just_confirmed:
                confirmed.append(st.just_confirmed)
        self.assertEqual(confirmed, ["A"])
        self.assertEqual(proc.get_text(), "A")

        # Release (no hand) for 10 frames unlocks re-signing 'A'
        for i in range(10):
            proc.update(None, 0.0, timestamp_sec=4.5 + i * 0.05)

        proc.insert_space()
        for i in range(15):
            proc.update("B", 0.90, timestamp_sec=5.5 + i * 0.1)
        self.assertEqual(proc.get_text(), "A B")

        proc.backspace()
        self.assertEqual(proc.get_text(), "A ")
        proc.clear()
        self.assertEqual(proc.get_text(), "")


class TestDeepMLP(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = FSLDeepMLP()
        model.eval()
        x = torch.randn(4, FEATURE_DIM)
        out = model(x)
        self.assertEqual(out.shape, (4, NUM_CLASSES))


if __name__ == "__main__":
    unittest.main()
