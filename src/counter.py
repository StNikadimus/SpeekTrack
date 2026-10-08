"""Speaking-event statistics, persisted separately from voice profiles."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.speaker_manager import UNKNOWN_NAME

FORMAT_VERSION = 1


class StatisticsStore:
    """Counts speaking events per person and saves them to a JSON file.

    The file only contains counts, so resetting statistics never touches the
    voice profiles in ``data/people``.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._counts: Dict[str, int] = {}
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.load()

    # ------------------------------------------------------------ persistence
    def load(self) -> None:
        if not self.path.exists():
            self._counts = {}
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            counts = data.get("counts", {})
            self._counts = {str(k): int(v) for k, v in counts.items() if int(v) > 0}
            self.created_at = data.get("created_at", self.created_at)
        except (ValueError, TypeError, AttributeError) as exc:
            # Keep the broken file for inspection instead of silently losing it.
            backup = self.path.with_name(self.path.name + ".corrupt")
            os.replace(self.path, backup)
            self._counts = {}
            print(f"Warning: could not read {self.path.name} ({exc}); "
                  f"moved it to {backup.name} and started fresh.")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": FORMAT_VERSION,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "counts": dict(sorted(self._counts.items())),
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    # --------------------------------------------------------------- counting
    def increment(self, name: str, save: bool = True) -> int:
        """Add one speaking event for ``name`` and return the new count."""
        self._counts[name] = self._counts.get(name, 0) + 1
        if save:
            self.save()
        return self._counts[name]

    def get(self, name: str) -> int:
        return self._counts.get(name, 0)

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def reset(self) -> None:
        self._counts = {}
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.save()

    def remove(self, name: str) -> None:
        if self._counts.pop(name, None) is not None:
            self.save()

    # ---------------------------------------------------------------- display
    def rows(self, registered: Iterable[str] = ()) -> List[Tuple[str, int]]:
        """(name, count) rows: people by count (desc), Unknown always last.

        Registered people without events are included with 0.
        """
        counts = {name: 0 for name in registered}
        counts.update(self._counts)
        unknown = counts.pop(UNKNOWN_NAME, 0)
        rows = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
        rows.append((UNKNOWN_NAME, unknown))
        return rows
