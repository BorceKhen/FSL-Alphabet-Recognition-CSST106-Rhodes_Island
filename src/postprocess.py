"""Post-processing pipelines for A-Z fingerspelling and FSL-105 dynamic phrase segmentation."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from src.config import (
    CONFIDENCE_THRESHOLD,
    DWELL_TIME_SECONDS,
    FSL105_SEQ_LEN,
    PHRASE_CONFIDENCE_THRESHOLD,
    PHRASE_COOLDOWN_SECONDS,
    PHRASE_IDLE_END_SECONDS,
    PHRASE_MAX_BUFFER_FRAMES,
    PHRASE_MIN_ACTIVE_FRAMES,
    SMOOTHING_WINDOW,
)
from src.features import resample_sequence


@dataclass
class ProcessorState:
    raw_label: Optional[str]
    raw_confidence: float
    smoothed_label: Optional[str]
    dwell_progress: float       # 0.0 to 1.0
    just_confirmed: Optional[str]
    text_buffer: str


class WordFormationProcessor:
    """Converts per-frame gesture predictions into confirmed characters and words."""

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        smoothing_window: int = SMOOTHING_WINDOW,
        dwell_time_seconds: float = DWELL_TIME_SECONDS,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.dwell_time_seconds = dwell_time_seconds

        self._window: Deque[Optional[str]] = deque(maxlen=smoothing_window)
        self._active_candidate: Optional[str] = None
        self._candidate_start_time: Optional[float] = None
        self._locked_label: Optional[str] = None
        self._buffer: list[str] = []

    def update(
        self,
        predicted_label: Optional[str],
        confidence: float,
        timestamp_sec: float,
    ) -> ProcessorState:
        """Process one frame's prediction and return updated UI/word state."""
        # Stage 1: Confidence Thresholding (tau = 0.75)
        valid_label = (
            predicted_label
            if (predicted_label is not None and confidence >= self.confidence_threshold)
            else None
        )
        self._window.append(valid_label)

        # Stage 2: Temporal Smoothing (10-frame sliding window majority vote)
        valid_votes = [lbl for lbl in self._window if lbl is not None]
        smoothed: Optional[str] = None
        if valid_votes:
            top_label, top_count = Counter(valid_votes).most_common(1)[0]
            # Require majority of the current window (or >= half of full window)
            if top_count >= max(1, len(self._window) // 2):
                smoothed = top_label

        # If hand is removed / below threshold, unlock so the same letter can be signed again
        if smoothed is None:
            self._active_candidate = None
            self._candidate_start_time = None
            self._locked_label = None
            return ProcessorState(
                raw_label=predicted_label,
                raw_confidence=confidence,
                smoothed_label=None,
                dwell_progress=0.0,
                just_confirmed=None,
                text_buffer=self.get_text(),
            )

        # If user already confirmed this held gesture, wait until they change or release
        if smoothed == self._locked_label:
            return ProcessorState(
                raw_label=predicted_label,
                raw_confidence=confidence,
                smoothed_label=smoothed,
                dwell_progress=1.0,
                just_confirmed=None,
                text_buffer=self.get_text(),
            )

        # Stage 3: Dwell-Time Confirmation (1.0 second steady hold)
        if smoothed != self._active_candidate:
            self._active_candidate = smoothed
            self._candidate_start_time = timestamp_sec
            self._locked_label = None

        elapsed = max(0.0, timestamp_sec - (self._candidate_start_time or timestamp_sec))
        progress = min(1.0, elapsed / max(self.dwell_time_seconds, 1e-6))
        just_confirmed: Optional[str] = None

        if elapsed >= self.dwell_time_seconds:
            self._buffer.append(smoothed)
            self._locked_label = smoothed
            just_confirmed = smoothed
            progress = 1.0

        return ProcessorState(
            raw_label=predicted_label,
            raw_confidence=confidence,
            smoothed_label=smoothed,
            dwell_progress=progress,
            just_confirmed=just_confirmed,
            text_buffer=self.get_text(),
        )

    def insert_space(self) -> str:
        if self._buffer and self._buffer[-1] != " ":
            self._buffer.append(" ")
        return self.get_text()

    def backspace(self) -> str:
        if self._buffer:
            self._buffer.pop()
        return self.get_text()

    def complete_current_word(self, word: str) -> str:
        """Replace trailing partial word after the last space with a completed word + trailing space."""
        clean = word.strip().upper()
        if not clean:
            return self.get_text()
        current = self.get_text()
        last_space = current.rfind(" ")
        prefix = current[: last_space + 1] if last_space != -1 else ""
        new_text = prefix + clean + " "
        self._buffer = list(new_text)
        return new_text

    def append_text(self, text: str) -> str:
        """Append arbitrary characters (e.g., quick phrase) to the text buffer."""
        if not text:
            return self.get_text()
        if self._buffer and self._buffer[-1] != " ":
            self._buffer.append(" ")
        self._buffer.extend(list(text))
        return self.get_text()

    def get_current_partial_word(self) -> str:
        """Return the trailing word fragment being fingerspelled."""
        text = self.get_text()
        if not text or text.endswith(" "):
            return ""
        return text.rsplit(" ", 1)[-1]

    def clear(self) -> str:
        self._buffer.clear()
        self._window.clear()
        self._active_candidate = None
        self._candidate_start_time = None
        self._locked_label = None
        return ""

    def get_text(self) -> str:
        return "".join(self._buffer)


@dataclass
class PhraseProcessorState:
    capture_state: str               # "idle" | "signing" | "finalized"
    active_frames: int
    progress: float                  # 0.0 to 1.0
    should_preview: bool
    ready_sequence: Optional[np.ndarray]
    tokens: List[Dict[str, Any]]


class PhraseSignProcessor:
    """Segments continuous webcam holistic frames into discrete FSL-105 sign sequences."""

    def __init__(
        self,
        confidence_threshold: float = PHRASE_CONFIDENCE_THRESHOLD,
        min_active_frames: int = PHRASE_MIN_ACTIVE_FRAMES,
        max_buffer_frames: int = PHRASE_MAX_BUFFER_FRAMES,
        idle_end_seconds: float = PHRASE_IDLE_END_SECONDS,
        cooldown_seconds: float = PHRASE_COOLDOWN_SECONDS,
        target_len: int = FSL105_SEQ_LEN,
        max_tokens: int = 30,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_active_frames = min_active_frames
        self.max_buffer_frames = max_buffer_frames
        self.idle_end_seconds = idle_end_seconds
        self.cooldown_seconds = cooldown_seconds
        self.target_len = target_len
        self.max_tokens = max_tokens

        self._active_vecs: List[np.ndarray] = []
        self._last_active_time: Optional[float] = None
        self._last_confirm_label: Optional[str] = None
        self._last_confirm_time: float = -999.0
        self._tokens: List[Dict[str, Any]] = []

    def update(
        self,
        vec_293: Optional[np.ndarray],
        hands_raised: bool,
        timestamp_sec: float,
        force_finalize: bool = False,
    ) -> PhraseProcessorState:
        """Accumulate active signing frames and emit a resampled sequence when sign completes."""
        if hands_raised and vec_293 is not None:
            self._active_vecs.append(vec_293.astype(np.float32))
            self._last_active_time = timestamp_sec
            n_frames = len(self._active_vecs)

            if force_finalize or n_frames >= self.max_buffer_frames:
                seq = resample_sequence(np.stack(self._active_vecs, axis=0), self.target_len)
                self._active_vecs.clear()
                self._last_active_time = None
                return PhraseProcessorState(
                    capture_state="finalized",
                    active_frames=n_frames,
                    progress=1.0,
                    should_preview=False,
                    ready_sequence=seq,
                    tokens=list(self._tokens),
                )

            should_preview = n_frames >= 10 and (n_frames % 4 == 0)
            preview_seq = (
                resample_sequence(np.stack(self._active_vecs, axis=0), self.target_len)
                if should_preview
                else None
            )
            progress = min(1.0, n_frames / float(max(self.min_active_frames, 1)))
            return PhraseProcessorState(
                capture_state="signing",
                active_frames=n_frames,
                progress=progress,
                should_preview=should_preview,
                ready_sequence=preview_seq,
                tokens=list(self._tokens),
            )

        # Hands lowered / idle
        n_frames = len(self._active_vecs)
        if n_frames > 0:
            idle_elapsed = (
                timestamp_sec - self._last_active_time
                if self._last_active_time is not None
                else 0.0
            )
            if force_finalize or idle_elapsed >= self.idle_end_seconds:
                if n_frames >= self.min_active_frames:
                    seq = resample_sequence(np.stack(self._active_vecs, axis=0), self.target_len)
                    self._active_vecs.clear()
                    self._last_active_time = None
                    return PhraseProcessorState(
                        capture_state="finalized",
                        active_frames=n_frames,
                        progress=1.0,
                        should_preview=False,
                        ready_sequence=seq,
                        tokens=list(self._tokens),
                    )
                # Too short: discard transient noise
                self._active_vecs.clear()
                self._last_active_time = None
            else:
                return PhraseProcessorState(
                    capture_state="signing",
                    active_frames=n_frames,
                    progress=min(1.0, n_frames / float(max(self.min_active_frames, 1))),
                    should_preview=False,
                    ready_sequence=None,
                    tokens=list(self._tokens),
                )

        return PhraseProcessorState(
            capture_state="idle",
            active_frames=0,
            progress=0.0,
            should_preview=False,
            ready_sequence=None,
            tokens=list(self._tokens),
        )

    def try_commit_prediction(
        self,
        top1: Dict[str, Any],
        timestamp_sec: float,
        bypass_threshold: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Commit finalized prediction to token composer if confident and outside cooldown."""
        prob = float(top1.get("prob", 0.0))
        label = str(top1.get("label", ""))
        if not bypass_threshold and prob < self.confidence_threshold:
            return None
        if (
            label == self._last_confirm_label
            and (timestamp_sec - self._last_confirm_time) < self.cooldown_seconds
        ):
            return None

        token = {
            "type": "phrase",
            "id": int(top1.get("id", -1)),
            "label": label,
            "category": str(top1.get("category", "")),
            "confidence": round(prob, 4),
        }
        self._tokens.append(token)
        if len(self._tokens) > self.max_tokens:
            self._tokens = self._tokens[-self.max_tokens :]
        self._last_confirm_label = label
        self._last_confirm_time = timestamp_sec
        return token

    def append_manual_token(
        self,
        label: str,
        class_id: int = -1,
        category: str = "",
        confidence: float = 1.0,
        token_type: str = "phrase",
    ) -> List[Dict[str, Any]]:
        self._tokens.append({
            "type": token_type,
            "id": int(class_id),
            "label": str(label),
            "category": str(category),
            "confidence": round(float(confidence), 4),
        })
        if len(self._tokens) > self.max_tokens:
            self._tokens = self._tokens[-self.max_tokens :]
        return list(self._tokens)

    def undo_token(self) -> List[Dict[str, Any]]:
        if self._tokens:
            self._tokens.pop()
        return list(self._tokens)

    def clear_tokens(self) -> List[Dict[str, Any]]:
        self._tokens.clear()
        self._last_confirm_label = None
        return []

    def reset_capture(self) -> None:
        self._active_vecs.clear()
        self._last_active_time = None

    def get_tokens(self) -> List[Dict[str, Any]]:
        return list(self._tokens)
