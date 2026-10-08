"""Shared fakes for tests - no microphone or neural network needed."""

import numpy as np

DIM = 256


def unit(seed: int) -> np.ndarray:
    """Deterministic random unit vector (a stand-in for a voice embedding)."""
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base: np.ndarray, noise: float, seed: int) -> np.ndarray:
    """Unit vector close to ``base`` (same 'speaker', different utterance)."""
    v = base + noise * np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEncoder:
    """Maps audio to an embedding by its (constant) sample value.

    Test audio for "speaker k" is filled with the value ``levels[k]``; the fake
    encoder looks the value up and returns that speaker's embedding.
    """

    def __init__(self, voices):
        self.voices = voices          # {level: embedding}
        self.calls = 0

    def embed(self, wav):
        self.calls += 1
        voiced = wav[np.abs(wav) > 0.05]
        if voiced.size == 0:
            return None
        level = round(float(np.median(voiced)), 2)
        return self.voices.get(level)


class FakeVAD:
    """Any frame whose absolute mean exceeds 0.05 is speech."""

    def is_speech(self, frame):
        return float(np.mean(np.abs(frame))) > 0.05
