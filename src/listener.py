"""Real-time pipeline: frames -> VAD -> speaking events -> speaker -> counter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np

from src.counter import StatisticsStore
from src.segmenter import READY, SpeechSegmenter
from src.speaker_recognition import Match, SpeakerIdentifier


@dataclass
class DetectedEvent:
    timestamp: datetime      # wall-clock time when the speech started
    match: Match
    count: int               # new event count of match.name
    voiced_seconds: float    # speech used for identification


class ListeningSession:
    """Feeds audio frames through the whole pipeline.

    It does not know where the frames come from, so it can be driven by the
    microphone (app.py) or by recorded audio / fakes (tests).

    ``vad`` needs ``is_speech(frame) -> bool`` and ``encoder`` needs
    ``embed(wav) -> np.ndarray | None``.
    """

    def __init__(self, vad, segmenter: SpeechSegmenter, encoder,
                 identifier: SpeakerIdentifier, stats: StatisticsStore,
                 frame_seconds: float, start_time: float):
        self.vad = vad
        self.segmenter = segmenter
        self.encoder = encoder
        self.identifier = identifier
        self.stats = stats
        self.frame_seconds = frame_seconds
        self.start_time = start_time
        self.events: List[DetectedEvent] = []
        self.speech_active = False

    def process_frame(self, frame: np.ndarray) -> List[DetectedEvent]:
        updates = self.segmenter.process(frame, self.vad.is_speech(frame))
        self.speech_active = self.segmenter.active
        return self._handle(updates)

    def finish(self) -> List[DetectedEvent]:
        """Count an utterance that is still in progress when listening stops."""
        return self._handle(self.segmenter.flush())

    def _handle(self, updates) -> List[DetectedEvent]:
        detected = []
        for update in updates:
            if update.kind != READY:
                continue
            match = self.identifier.identify(self.encoder.embed(update.audio))
            count = self.stats.increment(match.name)
            timestamp = datetime.fromtimestamp(
                self.start_time + update.start_frame * self.frame_seconds)
            event = DetectedEvent(timestamp, match, count, update.voiced_seconds)
            self.events.append(event)
            detected.append(event)
        return detected
