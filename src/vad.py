"""Frame-level voice activity detection (webrtcvad + energy gate)."""

from __future__ import annotations

import numpy as np

try:
    import webrtcvad  # provided by the "webrtcvad-wheels" package
except ImportError:  # pragma: no cover - depends on the environment
    webrtcvad = None

VALID_FRAME_MS = (10, 20, 30)
VALID_SAMPLE_RATES = (8000, 16000, 32000, 48000)


def frame_rms(frame: np.ndarray) -> float:
    """Root-mean-square level of a float frame (range -1..1)."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


def float_to_pcm16(frame: np.ndarray) -> bytes:
    """Convert float32 audio in -1..1 to little-endian 16-bit PCM bytes."""
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class VoiceActivityDetector:
    """Decides for each audio frame whether it contains speech.

    A frame counts as speech only if it is loud enough (RMS above
    ``energy_threshold``) AND webrtcvad classifies it as speech. The energy
    gate stops webrtcvad from firing on faint background noise.
    """

    def __init__(self, sample_rate: int, frame_ms: int, aggressiveness: int = 2,
                 energy_threshold: float = 0.0):
        if webrtcvad is None:
            raise ImportError(
                "webrtcvad is not installed. Run: pip install -r requirements.txt"
            )
        if sample_rate not in VALID_SAMPLE_RATES:
            raise ValueError(f"webrtcvad does not support {sample_rate} Hz")
        if frame_ms not in VALID_FRAME_MS:
            raise ValueError("FRAME_MS must be 10, 20 or 30")
        if not 0 <= aggressiveness <= 3:
            raise ValueError("VAD_AGGRESSIVENESS must be between 0 and 3")
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.energy_threshold = energy_threshold
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: np.ndarray) -> bool:
        if len(frame) != self.frame_samples:
            raise ValueError(
                f"expected {self.frame_samples} samples per frame, got {len(frame)}"
            )
        if frame_rms(frame) < self.energy_threshold:
            return False
        return self._vad.is_speech(float_to_pcm16(frame), self.sample_rate)

    def voiced_seconds(self, audio: np.ndarray) -> float:
        """Total duration of speech frames in a longer recording."""
        n_frames = len(audio) // self.frame_samples
        voiced = sum(
            self.is_speech(audio[i * self.frame_samples:(i + 1) * self.frame_samples])
            for i in range(n_frames)
        )
        return voiced * self.frame_samples / self.sample_rate
