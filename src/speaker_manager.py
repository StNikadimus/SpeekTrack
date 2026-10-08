"""Storage of registered people and their voice embeddings.

Layout on disk (one folder per person)::

    data/people/<slug>/profile.json    # name, timestamps, sample info
    data/people/<slug>/embeddings.npy  # float32 array, shape (n_samples, 256)

Every enrollment sample keeps its own embedding. Recognition uses the
L2-normalised mean of all samples (the "centroid").
"""

from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROFILE_FILE = "profile.json"
EMBEDDINGS_FILE = "embeddings.npy"
UNKNOWN_NAME = "Unknown"
MAX_NAME_LENGTH = 40


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify(name: str) -> str:
    """Folder-safe identifier: "Anže Novak" -> "anze-novak"."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("cannot normalise a zero vector")
    return (vector / norm).astype(np.float32)


@dataclass
class SpeakerProfile:
    name: str
    slug: str
    embeddings: np.ndarray                 # shape (n_samples, dim)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    sample_seconds: List[float] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def centroid(self) -> np.ndarray:
        return normalize(self.embeddings.mean(axis=0))


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class SpeakerManager:
    """Creates, loads, updates and removes speaker profiles on disk."""

    def __init__(self, people_dir: Path):
        self.people_dir = Path(people_dir)
        self.people_dir.mkdir(parents=True, exist_ok=True)
        self.load_errors: List[str] = []

    # ------------------------------------------------------------- validation
    def validate_name(self, name: str) -> Optional[str]:
        """Return an error message, or None if the name can be used."""
        name = name.strip()
        if not name:
            return "Name cannot be empty."
        if len(name) > MAX_NAME_LENGTH:
            return f"Name is too long (max {MAX_NAME_LENGTH} characters)."
        if name.lower() == UNKNOWN_NAME.lower():
            return f'"{UNKNOWN_NAME}" is reserved for unrecognised speakers.'
        if not slugify(name):
            return "Name must contain at least one letter or digit."
        return None

    # ---------------------------------------------------------------- reading
    def _folder(self, name: str) -> Path:
        return self.people_dir / slugify(name)

    def _load_folder(self, folder: Path) -> SpeakerProfile:
        meta = json.loads((folder / PROFILE_FILE).read_text(encoding="utf-8"))
        embeddings = np.load(folder / EMBEDDINGS_FILE).astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(f"unexpected embedding shape {embeddings.shape}")
        return SpeakerProfile(
            name=meta["name"],
            slug=folder.name,
            embeddings=embeddings,
            created_at=meta.get("created_at", _now()),
            updated_at=meta.get("updated_at", _now()),
            sample_seconds=list(meta.get("sample_seconds", [])),
        )

    def list_profiles(self) -> List[SpeakerProfile]:
        """All valid profiles, sorted by name. Broken folders are reported in
        ``load_errors`` instead of crashing the program."""
        self.load_errors = []
        profiles = []
        for folder in sorted(p for p in self.people_dir.iterdir() if p.is_dir()):
            if not (folder / PROFILE_FILE).exists():
                continue
            try:
                profiles.append(self._load_folder(folder))
            except Exception as exc:  # corrupted / partially written profile
                self.load_errors.append(f"{folder.name}: {exc}")
        return sorted(profiles, key=lambda p: p.name.lower())

    def get(self, name: str) -> Optional[SpeakerProfile]:
        folder = self._folder(name)
        if not (folder / PROFILE_FILE).exists():
            return None
        return self._load_folder(folder)

    def exists(self, name: str) -> bool:
        return bool(slugify(name)) and (self._folder(name) / PROFILE_FILE).exists()

    def centroids(self) -> Dict[str, np.ndarray]:
        return {p.name: p.centroid for p in self.list_profiles()}

    # ---------------------------------------------------------------- writing
    def save(self, name: str, embeddings: np.ndarray, sample_seconds: List[float],
             append: bool = False) -> SpeakerProfile:
        """Create (or replace / extend) a profile and write it to disk."""
        error = self.validate_name(name)
        if error:
            raise ValueError(error)
        name = name.strip()
        embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if embeddings.shape[0] == 0:
            raise ValueError("at least one embedding is required")

        existing = self.get(name)
        created_at = existing.created_at if existing else _now()
        if append and existing is not None:
            if existing.embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError("embedding size does not match the existing profile")
            embeddings = np.vstack([existing.embeddings, embeddings])
            sample_seconds = existing.sample_seconds + list(sample_seconds)
            name = existing.name

        profile = SpeakerProfile(name=name, slug=slugify(name), embeddings=embeddings,
                                 created_at=created_at, updated_at=_now(),
                                 sample_seconds=[round(float(s), 2) for s in sample_seconds])
        folder = self.people_dir / profile.slug
        folder.mkdir(parents=True, exist_ok=True)

        tmp_npy = folder / (EMBEDDINGS_FILE + ".tmp.npy")
        np.save(tmp_npy, profile.embeddings)
        os.replace(tmp_npy, folder / EMBEDDINGS_FILE)
        meta = {
            "name": profile.name,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "num_samples": profile.num_samples,
            "sample_seconds": profile.sample_seconds,
            "embedding_dim": int(profile.embeddings.shape[1]),
            "embedding_model": "resemblyzer.VoiceEncoder (pretrained.pt)",
        }
        _atomic_write_text(folder / PROFILE_FILE,
                           json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        return profile

    def remove(self, name: str) -> bool:
        folder = self._folder(name)
        if not slugify(name) or not folder.is_dir():
            return False
        shutil.rmtree(folder)
        return True
