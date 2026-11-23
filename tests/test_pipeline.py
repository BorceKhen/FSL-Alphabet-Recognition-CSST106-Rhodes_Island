"""Unit and integration tests for A-Z alphabet, FSL-105 phrase, translator, and Web API."""

from __future__ import annotations

import base64
import unittest
from pathlib import Path

import numpy as np
import torch

from src.config import (
    FEATURE_DIM,
    FSL105_DIR,
    FSL105_FEATURE_DIM,
    FSL105_MODEL_PATH,
    FSL105_NUM_CLASSES,
    FSL105_SEQ_LEN,
    NUM_CLASSES,
)
from src.features import normalize_holistic_landmarks, normalize_landmarks
from src.postprocess import PhraseSignProcessor, WordFormationProcessor
from src.train_fsl105 import FSLPhraseNet
from src.train_mlp import FSLDeepMLP
from src.translator import (
    autocomplete_word,
    clean_fsl_label,
    compose_sentence,
    match_listener_signs,
)


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
        np.testing.assert_allclose(v_base[:3], np.zeros(3), atol=1e-6)

    def test_holistic_normalization_invariance_and_masking(self) -> None:
        rng = np.random.default_rng(7)
        pose = rng.uniform(0.2, 0.8, size=(15, 3)).astype(np.float32)
        # Ensure shoulders (7 and 8) have span >= 0.25 so scale clamping doesn't trigger
        pose[7] = np.array([0.35, 0.45, 0.0], dtype=np.float32)
        pose[8] = np.array([0.65, 0.45, 0.0], dtype=np.float32)
        rh = rng.uniform(0.3, 0.7, size=(21, 3)).astype(np.float32)

        vec1 = normalize_holistic_landmarks(pose, None, rh)
        self.assertEqual(vec1.shape, (FSL105_FEATURE_DIM,))
        # Left hand absent -> lh_body (45:108) and lh_local (171:231) and flag[0] must be 0
        np.testing.assert_allclose(vec1[45:108], 0.0, atol=1e-6)
        np.testing.assert_allclose(vec1[171:231], 0.0, atol=1e-6)
        self.assertEqual(vec1[291], 0.0)
        self.assertEqual(vec1[292], 1.0)

        # Translation + uniform scaling around origin should preserve normalized vector
        shift = np.array([1.2, -0.7, 0.3], dtype=np.float32)
        vec2 = normalize_holistic_landmarks((pose + shift) * 2.0, None, (rh + shift) * 2.0)
        np.testing.assert_allclose(vec1, vec2, atol=1e-4)


class TestWordFormationProcessor(unittest.TestCase):
    def test_threshold_smoothing_and_dwell_confirmation(self) -> None:
        proc = WordFormationProcessor(
            confidence_threshold=0.75,
            smoothing_window=10,
            dwell_time_seconds=1.0,
        )

        for i in range(15):
            st = proc.update("A", 0.50, timestamp_sec=i * 0.1)
            self.assertIsNone(st.just_confirmed)
        self.assertEqual(proc.get_text(), "")

        confirmed = []
        for i in range(20):
            st = proc.update("A", 0.92, timestamp_sec=2.0 + i * 0.1)
            if st.just_confirmed:
                confirmed.append(st.just_confirmed)
        self.assertEqual(confirmed, ["A"])
        self.assertEqual(proc.get_text(), "A")

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


class TestPhraseSignProcessor(unittest.TestCase):
    def test_idle_signing_finalize_and_cooldown(self) -> None:
        proc = PhraseSignProcessor(
            confidence_threshold=0.55,
            min_active_frames=8,
            idle_end_seconds=0.5,
            cooldown_seconds=1.5,
        )
        dummy = np.ones(FSL105_FEATURE_DIM, dtype=np.float32)

        st0 = proc.update(dummy, hands_raised=False, timestamp_sec=0.0)
        self.assertEqual(st0.capture_state, "idle")

        for i in range(10):
            st = proc.update(dummy, hands_raised=True, timestamp_sec=0.1 * (i + 1))
            self.assertEqual(st.capture_state, "signing")

        # Brief pause < 0.5s stays in signing
        st_wait = proc.update(dummy, hands_raised=False, timestamp_sec=1.2)
        self.assertEqual(st_wait.capture_state, "signing")

        # Pause >= 0.5s finalizes and emits (40, 293) sequence
        st_fin = proc.update(dummy, hands_raised=False, timestamp_sec=1.6)
        self.assertEqual(st_fin.capture_state, "finalized")
        self.assertIsNotNone(st_fin.ready_sequence)
        self.assertEqual(st_fin.ready_sequence.shape, (FSL105_SEQ_LEN, FSL105_FEATURE_DIM))

        tok = proc.try_commit_prediction(
            {"id": 0, "label": "GOOD MORNING", "category": "GREETING", "prob": 0.91},
            timestamp_sec=1.6,
        )
        self.assertIsNotNone(tok)
        self.assertEqual(len(proc.get_tokens()), 1)

        # Duplicate within cooldown window should be suppressed
        dup = proc.try_commit_prediction(
            {"id": 0, "label": "GOOD MORNING", "category": "GREETING", "prob": 0.90},
            timestamp_sec=2.0,
        )
        self.assertIsNone(dup)

        proc.undo_token()
        self.assertEqual(len(proc.get_tokens()), 0)


class TestModelsAndTranslator(unittest.TestCase):
    def test_deep_mlp_forward_shape(self) -> None:
        model = FSLDeepMLP()
        model.eval()
        x = torch.randn(4, FEATURE_DIM)
        out = model(x)
        self.assertEqual(out.shape, (4, NUM_CLASSES))

    def test_phrase_net_forward_shape(self) -> None:
        model = FSLPhraseNet()
        model.eval()
        x = torch.randn(2, FSL105_SEQ_LEN, FSL105_FEATURE_DIM)
        out = model(x)
        self.assertEqual(out.shape, (2, FSL105_NUM_CLASSES))

    def test_translator_cleanup_compose_autocomplete_and_matching(self) -> None:
        self.assertEqual(clean_fsl_label("NINE "), "NINE")
        self.assertEqual(clean_fsl_label("WEELCHAIR PERSON"), "WHEELCHAIR PERSON")
        self.assertEqual(clean_fsl_label("DON’T UNDERSTAND"), "DON'T UNDERSTAND")
        self.assertEqual(clean_fsl_label("IM FINE"), "I'M FINE")

        en_single = compose_sentence([{"type": "phrase", "label": "DON'T UNDERSTAND"}], lang="en")
        fil_single = compose_sentence([{"type": "phrase", "label": "DON'T UNDERSTAND"}], lang="fil")
        self.assertIn("understand", en_single.lower())
        self.assertIn("maintindihan", fil_single.lower())

        en_multi = compose_sentence(
            [{"type": "phrase", "label": "HOT"}, {"type": "phrase", "label": "COFFEE"}, {"type": "phrase", "label": "NO SUGAR"}],
            lang="en",
        )
        self.assertIn("hot coffee", en_multi.lower())
        self.assertIn("no sugar", en_multi.lower())

        fil_multi = compose_sentence(
            [{"type": "phrase", "label": "HOT"}, {"type": "phrase", "label": "COFFEE"}],
            lang="fil",
        )
        self.assertIn("mainit na kape", fil_multi.lower())

        sugg = autocomplete_word("PHARM")
        self.assertIn("PHARMACY", sugg)

        matched = match_listener_signs("Good morning, would you like coffee with no sugar?")
        labels = [m["label"] for m in matched]
        self.assertIn("GOOD MORNING", labels)
        self.assertIn("COFFEE", labels)
        self.assertIn("NO SUGAR", labels)


class TestWebApiSmoke(unittest.TestCase):
    def test_web_routes_and_two_way_chat(self) -> None:
        from src.web_app import app

        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        cat = client.get("/api/fsl105/catalog")
        self.assertEqual(cat.status_code, 200)

        # A-Z static reference image prediction
        ref_b64 = base64.b64encode(Path("data/reference/B/0.jpg").read_bytes()).decode("ascii")
        res_az = client.post("/api/predict", json={"image": f"data:image/jpeg;base64,{ref_b64}"})
        self.assertEqual(res_az.status_code, 200)
        self.assertIn("top3", res_az.get_json())

        # Composer & Two-way listener reply
        client.post("/api/composer/clear")
        client.post("/api/composer/reset_chat")
        pick = client.post("/api/composer/pick_phrase", json={"label": "GOOD MORNING"})
        self.assertEqual(pick.status_code, 200)
        sent = client.post("/api/composer/send", json={"mode": "phrase"})
        self.assertEqual(sent.status_code, 200)
        self.assertEqual(len(sent.get_json()["conversation"]), 1)

        reply = client.post("/api/composer/listener_reply", json={"text": "Good morning! Thank you."})
        self.assertEqual(reply.status_code, 200)
        rjson = reply.get_json()
        self.assertEqual(len(rjson["conversation"]), 2)
        self.assertGreaterEqual(len(rjson["message"]["matched_signs"]), 2)

        # If FSL-105 model is trained, test real held-out clip prediction endpoint
        sample_clip = FSL105_DIR / "clips" / "0" / "17.MOV"
        if FSL105_MODEL_PATH.exists() and sample_clip.exists():
            vid_res = client.post(
                "/api/predict_phrase_video",
                json={"sample_path": "clips/0/17.MOV", "append": False},
            )
            self.assertEqual(vid_res.status_code, 200)
            self.assertIn("top5", vid_res.get_json())


if __name__ == "__main__":
    unittest.main()
