"""Flask web UI server for FSL Alphabet Recognition & Word Formation."""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from src.app import ModelPredictor
from src.config import CLASSES, CONFIDENCE_THRESHOLD, SPACE_PAUSE_SECONDS
from src.features import HAND_CONNECTIONS, HandLandmarkExtractor
from src.postprocess import WordFormationProcessor

BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

_lock = threading.Lock()
_predictor = ModelPredictor()
_extractor = HandLandmarkExtractor(running_mode="IMAGE")
_processor = WordFormationProcessor()
_last_hand_time: float | None = None
_auto_spaced = False


def _decode_frame(data_url: str) -> np.ndarray | None:
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template(
        "index.html",
        classes=CLASSES,
        has_svm=_predictor.svm is not None,
        confidence_threshold=int(CONFIDENCE_THRESHOLD * 100),
        hand_connections=[list(pair) for pair in HAND_CONNECTIONS],
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    global _last_hand_time, _auto_spaced
    payload = request.get_json(silent=True) or {}
    frame_b64 = payload.get("image")
    model_mode = payload.get("model", "Deep MLP")
    use_clahe = bool(payload.get("clahe", False))
    auto_space = bool(payload.get("auto_space", False))

    if not frame_b64:
        return jsonify({"error": "Missing image payload"}), 400

    frame = _decode_frame(frame_b64)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    t0 = time.perf_counter()
    now_mono = time.monotonic()
    with _lock:
        vec, landmarks = _extractor.extract(frame, use_clahe=use_clahe)
        top3 = []
        raw_label = None
        raw_conf = 0.0
        lm_points = []

        if vec is not None and landmarks is not None:
            _last_hand_time = now_mono
            _auto_spaced = False
            top3 = [
                {"label": lbl, "prob": round(prob, 4)}
                for lbl, prob in _predictor.predict_top_k(vec, mode=model_mode, k=3)
            ]
            raw_label = top3[0]["label"]
            raw_conf = top3[0]["prob"]
            lm_points = [[round(float(lm.x), 4), round(float(lm.y), 4)] for lm in landmarks]
        elif (
            auto_space
            and _last_hand_time is not None
            and not _auto_spaced
            and (now_mono - _last_hand_time) >= SPACE_PAUSE_SECONDS
        ):
            _processor.insert_space()
            _auto_spaced = True

        state = _processor.update(raw_label, raw_conf, now_mono)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return jsonify({
        "hand_detected": vec is not None,
        "landmarks": lm_points,
        "top3": top3,
        "raw_label": state.raw_label,
        "raw_confidence": round(state.raw_confidence, 4),
        "smoothed_label": state.smoothed_label,
        "dwell_progress": round(state.dwell_progress, 4),
        "just_confirmed": state.just_confirmed,
        "text_buffer": state.text_buffer,
        "latency_ms": latency_ms,
    })


@app.route("/api/buffer/<action>", methods=["POST"])
def api_buffer(action: str):
    with _lock:
        if action == "space":
            text = _processor.insert_space()
        elif action == "backspace":
            text = _processor.backspace()
        elif action == "clear":
            text = _processor.clear()
        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400
    return jsonify({"text_buffer": text})


@app.route("/api/reference/<letter>")
def api_reference(letter: str):
    letter = letter.upper()
    if letter not in CLASSES:
        return jsonify({"error": "Invalid letter"}), 404

    idx = max(0, request.args.get("idx", default=0, type=int)) % 4
    bundled_img = BASE_DIR / "static" / "reference" / letter / f"{idx}.jpg"
    if not bundled_img.exists():
        return jsonify({"error": "Reference image not found"}), 404
    return send_file(bundled_img, mimetype="image/jpeg")


def main() -> None:
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
