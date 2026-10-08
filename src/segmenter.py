"""Turns a stream of (frame, is_speech) pairs into speaking events.

A speaking event starts when enough voiced frames appear in a short window
and ends after ``pause_seconds`` of continuous silence. Short pauses inside an
utterance (breathing, between words) therefore do not split the event.

For every event the segmenter emits:

* exactly one ``"ready"`` update carrying audio for speaker identification -
  as soon as ``identify_after_seconds`` of speech has been collected, or when
  the event ends if it is shorter. Events with less than
  ``min_event_seconds`` of speech never become ready (noise, clicks).
* exactly one ``"ended"`` update when the pause threshold is reached.

This module is pure logic (no audio device, no model) so it is easy to test.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

READY = "ready"
ENDED = "ended"


@dataclass
class SegmentUpdate:
    kind: str                     # READY or ENDED
    event_index: int              # 1, 2, 3 ... within this segmenter
    start_frame: int              # global index of the first voiced frame
    voiced_seconds: float         # speech collected so far (READY) / in total (ENDED)
    audio: Optional[np.ndarray] = None   # only for READY
    counted: bool = False         # only for ENDED: did this event become READY?


def _to_frames(seconds: float, frame_seconds: float) -> int:
    return max(1, int(round(seconds / frame_seconds)))


class SpeechSegmenter:
    def __init__(self, frame_seconds: float, pause_seconds: float,
                 min_start_seconds: float, min_event_seconds: float,
                 identify_after_seconds: float, pre_padding_seconds: float):
        self.frame_seconds = frame_seconds
        self.pause_frames = _to_frames(pause_seconds, frame_seconds)
        self.start_frames = _to_frames(min_start_seconds, frame_seconds)
        # Speech onsets flicker in webrtcvad, so we look for start_frames voiced
        # frames inside a window twice that long instead of a strict run.
        self.start_window_frames = self.start_frames * 2
        self.min_event_frames = _to_frames(min_event_seconds, frame_seconds)
        self.identify_frames = max(_to_frames(identify_after_seconds, frame_seconds),
                                   self.min_event_frames)
        pre_pad_frames = int(round(pre_padding_seconds / frame_seconds))

        self._history = deque(maxlen=pre_pad_frames + self.start_window_frames)
        self._frame_index = 0
        self._event_count = 0
        self._reset_event()

    # ------------------------------------------------------------------ state
    def _reset_event(self) -> None:
        self.active = False
        self._ready_sent = False
        self._frames: List[np.ndarray] = []
        self._last_voiced_len = 0
        self._voiced = 0
        self._silence_run = 0
        self._start_frame = 0

    @property
    def event_count(self) -> int:
        """Number of events that became READY (i.e. were counted)."""
        return self._event_count

    def _seconds(self, frames: int) -> float:
        return frames * self.frame_seconds

    def _trimmed_audio(self) -> np.ndarray:
        frames = self._frames[:self._last_voiced_len] or self._frames
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)

    def _make_ready(self) -> SegmentUpdate:
        self._ready_sent = True
        self._event_count += 1
        update = SegmentUpdate(READY, self._event_count, self._start_frame,
                               self._seconds(self._voiced), audio=self._trimmed_audio())
        # The audio is no longer needed once the speaker has been identified.
        self._frames = []
        self._last_voiced_len = 0
        return update

    def _end_event(self) -> List[SegmentUpdate]:
        updates: List[SegmentUpdate] = []
        if not self._ready_sent and self._voiced >= self.min_event_frames:
            updates.append(self._make_ready())
        updates.append(SegmentUpdate(ENDED, self._event_count if self._ready_sent else 0,
                                     self._start_frame, self._seconds(self._voiced),
                                     counted=self._ready_sent))
        self._reset_event()
        self._history.clear()
        return updates

    # ------------------------------------------------------------------- API
    def process(self, frame: np.ndarray, is_speech: bool) -> List[SegmentUpdate]:
        index = self._frame_index
        self._frame_index += 1

        if not self.active:
            self._history.append((index, frame, is_speech))
            recent = list(self._history)[-self.start_window_frames:]
            if sum(flag for _, _, flag in recent) < self.start_frames:
                return []
            # Open a new event, including the padding before the onset.
            self.active = True
            history = list(self._history)
            self._history.clear()
            self._frames = [f for _, f, _ in history]
            self._voiced = sum(flag for _, _, flag in history)
            self._last_voiced_len = max(i + 1 for i, (_, _, flag) in enumerate(history) if flag)
            self._start_frame = next(idx for idx, _, flag in history if flag)
        else:
            if not self._ready_sent:
                self._frames.append(frame)
            if is_speech:
                self._voiced += 1
                self._silence_run = 0
                self._last_voiced_len = len(self._frames)
            else:
                self._silence_run += 1

        updates: List[SegmentUpdate] = []
        if not self._ready_sent and self._voiced >= self.identify_frames:
            updates.append(self._make_ready())
        if self._silence_run >= self.pause_frames:
            updates.extend(self._end_event())
        return updates

    def flush(self) -> List[SegmentUpdate]:
        """Close an event that is still open (e.g. when listening stops)."""
        if not self.active:
            return []
        return self._end_event()
