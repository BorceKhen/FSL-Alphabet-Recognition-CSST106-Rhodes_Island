"""Post-processing pipeline: confidence filtering, 10-frame smoothing, and dwell-time word formation."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional

from src.config import CONFIDENCE_THRESHOLD, DWELL_TIME_SECONDS, SMOOTHING_WINDOW


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

    def clear(self) -> str:
        self._buffer.clear()
        self._window.clear()
        self._active_candidate = None
        self._candidate_start_time = None
        self._locked_label = None
        return ""

    def get_text(self) -> str:
        return "".join(self._buffer)
