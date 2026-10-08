"""Recording voice samples for a new person, with a live speaker check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from src.speaker_manager import normalize
from src.speaker_recognition import cosine_similarity

BAR_WIDTH = 20


def progress_bar(fraction: float, width: int = BAR_WIDTH) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


@dataclass
class LiveStatus:
    voice_detected: bool
    speaker: str                         # what to show after "Current speaker:"
    similarity: Optional[float] = None
    matches_target: bool = False


class LiveSpeakerCheck:
    """Decides, during enrollment, whether the current voice is the person
    being recorded.

    The first voiced windows build a reference embedding for the new person;
    every later window is compared with that reference and with the people
    who are already registered.
    """

    def __init__(self, encoder, target_name: str, others: Dict[str, np.ndarray],
                 match_threshold: float, other_threshold: float,
                 reference_windows: int = 2,
                 initial_reference: Optional[List[np.ndarray]] = None):
        self.encoder = encoder
        self.target_name = target_name
        self.others = others
        self.match_threshold = match_threshold
        self.other_threshold = other_threshold
        self.reference_windows = reference_windows
        # When adding samples to an existing person, their saved profile is
        # the reference from the start.
        self._reference: List[np.ndarray] = list(initial_reference or [])
        if self._reference:
            self.reference_windows = 0
        self.matching_windows = 0
        self.mismatching_windows = 0
        self.registered_matches: Dict[str, int] = {}   # windows that sounded like someone else

    def check(self, window: np.ndarray, voice_detected: bool) -> LiveStatus:
        if not voice_detected:
            return LiveStatus(False, "Silence")
        embedding = self.encoder.embed(window)
        if embedding is None:
            return LiveStatus(True, "(too little speech to tell)")

        other_name, other_sim = None, -1.0
        for name, centroid in self.others.items():
            sim = cosine_similarity(embedding, centroid)
            if sim > other_sim:
                other_name, other_sim = name, sim
        other_hit = other_name is not None and other_sim >= self.other_threshold

        if len(self._reference) < self.reference_windows:
            if other_hit:
                return self._registered_match(other_name, other_sim)
            self._reference.append(embedding)
            self.matching_windows += 1
            return LiveStatus(True, f"{self.target_name} (building voice reference...)",
                              matches_target=True)

        reference = normalize(np.mean(self._reference, axis=0))
        sim = cosine_similarity(embedding, reference)
        if other_hit and other_sim > sim:
            return self._registered_match(other_name, other_sim)
        if sim >= self.match_threshold:
            self._reference.append(embedding)
            self.matching_windows += 1
            return LiveStatus(True, self.target_name, sim, matches_target=True)
        self.mismatching_windows += 1
        return LiveStatus(True, "Different voice?", sim)


    def _registered_match(self, name: str, similarity: float) -> LiveStatus:
        self.mismatching_windows += 1
        self.registered_matches[name] = self.registered_matches.get(name, 0) + 1
        return LiveStatus(True, f"{name} (already registered!)", similarity)


def sample_consistency(embeddings: List[np.ndarray]) -> List[float]:
    """For each sample, its similarity to the most similar *other* sample.

    Using the best pairwise match (instead of the average of the others) means
    one bad sample is reported on its own without dragging down the scores of
    the good ones.
    """
    if len(embeddings) < 2:
        return [1.0] * len(embeddings)
    return [max(cosine_similarity(emb, other)
                for j, other in enumerate(embeddings) if j != i)
            for i, emb in enumerate(embeddings)]


@dataclass
class RecordedSample:
    audio: np.ndarray
    voiced_seconds: float
    matching_windows: int
    mismatching_windows: int
    registered_matches: Dict[str, int]   # other registered people heard in this sample


def record_sample(mic, vad, live_check: LiveSpeakerCheck, duration: float,
                  frame_seconds: float, check_interval: float, check_window: float,
                  out: Callable[[str], None] = print) -> RecordedSample:
    """Record ``duration`` seconds from an open microphone, printing a live
    status line every ``check_interval`` seconds."""
    total_frames = int(round(duration / frame_seconds))
    frames_per_check = max(1, int(round(check_interval / frame_seconds)))
    next_check = check_interval
    window_frames = max(1, int(round(check_window / frame_seconds)))

    frames: List[np.ndarray] = []
    flags: List[bool] = []
    matching_before = live_check.matching_windows
    mismatching_before = live_check.mismatching_windows
    registered_before = dict(live_check.registered_matches)
    while len(frames) < total_frames:
        frame = mic.read_frame(timeout=2.0)
        if frame is None:
            raise RuntimeError("The microphone stopped delivering audio.")
        frames.append(frame)
        flags.append(vad.is_speech(frame))

        elapsed = len(frames) * frame_seconds
        if elapsed >= next_check - frame_seconds / 2:
            next_check += check_interval
            recent_flags = flags[-frames_per_check:]
            voice = sum(recent_flags) >= max(1, 0.3 * len(recent_flags))
            # Analyse only the speech frames of the last few seconds, so a
            # window that is mostly silence does not give a misleading result.
            recent = zip(frames[-2 * window_frames:], flags[-2 * window_frames:])
            voiced = [f for f, is_voiced in recent if is_voiced][-window_frames:]
            if voice and len(voiced) < window_frames // 2:
                status = LiveStatus(True, "(too little speech to tell yet)")
            else:
                window = np.concatenate(voiced) if voiced else np.zeros(0, np.float32)
                status = live_check.check(window, voice)
            similarity = f" ({status.similarity:.2f})" if status.similarity is not None else ""
            out(f"{progress_bar(len(frames) / total_frames)} {elapsed:4.1f}/{duration:g}s  "
                f"Voice detected: {'YES' if status.voice_detected else 'NO '}  "
                f"Current speaker: {status.speaker}{similarity}")

    return RecordedSample(
        audio=np.concatenate(frames),
        voiced_seconds=sum(flags) * frame_seconds,
        matching_windows=live_check.matching_windows - matching_before,
        mismatching_windows=live_check.mismatching_windows - mismatching_before,
        registered_matches={name: n - registered_before.get(name, 0)
                            for name, n in live_check.registered_matches.items()
                            if n > registered_before.get(name, 0)},
    )
