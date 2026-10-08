"""Speaker embeddings (Resemblyzer) and matching against registered people.

Resemblyzer's ``VoiceEncoder`` turns an utterance into a 256-dimensional,
L2-normalised voice embedding. Two utterances of the same person give
embeddings with a high cosine similarity (dot product); different people give
lower values. A segment is attributed to the most similar registered person
only if that similarity clears a threshold and is clearly better than the
runner-up - otherwise the result is "Unknown".
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from src.speaker_manager import UNKNOWN_NAME, normalize

# Resemblyzer pads anything shorter than one 1.6 s window with zeros; below
# this amount of speech the embedding is mostly padding and not usable.
MIN_EMBED_SECONDS = 0.2
# Slightly longer than the encoder's 1.6 s analysis window.
MIN_WINDOW_SECONDS = 1.7


def _import_resemblyzer():
    """Import Resemblyzer 0.1.4, silencing a harmless SciPy deprecation.

    Resemblyzer imports ``scipy.ndimage.morphology`` which SciPy deprecated
    (still present in all SciPy 1.x releases, hence ``scipy<2`` in
    requirements.txt).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import resemblyzer
    return resemblyzer


class SpeakerEncoder:
    """Lazy wrapper around ``resemblyzer.VoiceEncoder``."""

    def __init__(self, device: Optional[str] = None):
        resemblyzer = _import_resemblyzer()
        self._preprocess_wav = resemblyzer.preprocess_wav
        self.sample_rate = resemblyzer.sampling_rate
        # The bundled pretrained.pt loads fine with PyTorch's safe
        # (weights_only) loader, the default since PyTorch 2.6.
        self._encoder = resemblyzer.VoiceEncoder(device=device, verbose=False)

    def embed(self, wav: np.ndarray) -> Optional[np.ndarray]:
        """Embedding of a 16 kHz float32 waveform, or None if it has no usable speech."""
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) < MIN_EMBED_SECONDS * self.sample_rate or not np.any(wav):
            return None
        # Volume normalisation + removal of long silences (Resemblyzer's own
        # preprocessing, which the model was trained with).
        processed = self._preprocess_wav(wav)
        if len(processed) < MIN_EMBED_SECONDS * self.sample_rate:
            processed = wav  # its internal VAD removed nearly everything; use the raw segment
        # The encoder looks at 1.6 s windows and zero-pads shorter input, which
        # hurts short utterances. Repeating the speech instead measurably
        # improves recognition of ~0.5-1.5 s segments.
        min_len = int(MIN_WINDOW_SECONDS * self.sample_rate)
        if len(processed) < min_len:
            processed = np.tile(processed, int(np.ceil(min_len / len(processed))))[:min_len]
        embedding = self._encoder.embed_utterance(processed)
        if not np.all(np.isfinite(embedding)):
            return None
        return embedding.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class Match:
    """Result of comparing one speech segment with all registered people."""
    name: str                       # person name or UNKNOWN_NAME
    similarity: float               # similarity to the best candidate
    best_candidate: Optional[str]   # closest person even if rejected
    reason: str = ""                # why the result is Unknown (empty if known)
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def is_known(self) -> bool:
        return self.name != UNKNOWN_NAME


class SpeakerIdentifier:
    """Matches embeddings against person centroids with an open-set decision."""

    def __init__(self, centroids: Dict[str, np.ndarray], threshold: float,
                 min_margin: float = 0.0):
        self.threshold = threshold
        self.min_margin = min_margin
        self.update_profiles(centroids)

    def update_profiles(self, centroids: Dict[str, np.ndarray]) -> None:
        self.centroids = {name: normalize(np.asarray(c, dtype=np.float32))
                          for name, c in centroids.items()}

    def scores(self, embedding: np.ndarray) -> Dict[str, float]:
        return {name: cosine_similarity(embedding, c) for name, c in self.centroids.items()}

    def identify(self, embedding: Optional[np.ndarray]) -> Match:
        if embedding is None:
            return Match(UNKNOWN_NAME, 0.0, None, reason="no usable speech")
        if not self.centroids:
            return Match(UNKNOWN_NAME, 0.0, None, reason="no registered people")

        scores = self.scores(embedding)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_name, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0

        if best < self.threshold:
            return Match(UNKNOWN_NAME, best, best_name, scores=scores,
                         reason=f"similarity {best:.2f} < threshold {self.threshold:.2f}")
        if best - second < self.min_margin:
            return Match(UNKNOWN_NAME, best, best_name, scores=scores,
                         reason=f"too close to {ranked[1][0]} ({second:.2f})")
        return Match(best_name, best, best_name, scores=scores)
