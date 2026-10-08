import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.counter import StatisticsStore
from src.speaker_manager import SpeakerManager
from tests.helpers import unit


class StatisticsStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.path = self.data_dir / "statistics.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_increment_and_get(self):
        stats = StatisticsStore(self.path)
        self.assertEqual(stats.increment("Nik"), 1)
        self.assertEqual(stats.increment("Nik"), 2)
        self.assertEqual(stats.increment("Unknown"), 1)
        self.assertEqual(stats.get("Nik"), 2)
        self.assertEqual(stats.get("Luka"), 0)
        self.assertEqual(stats.total, 3)

    def test_statistics_survive_restart(self):
        stats = StatisticsStore(self.path)
        for _ in range(3):
            stats.increment("Anže")
        stats.increment("Unknown")

        reloaded = StatisticsStore(self.path)
        self.assertEqual(reloaded.counts, {"Anže": 3, "Unknown": 1})
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["counts"]["Anže"], 3)

    def test_reset_keeps_voice_profiles(self):
        manager = SpeakerManager(self.data_dir / "people")
        manager.save("Nik", np.stack([unit(1)]), [5])
        stats = StatisticsStore(self.path)
        stats.increment("Nik")

        stats.reset()

        self.assertEqual(StatisticsStore(self.path).counts, {})
        self.assertTrue(SpeakerManager(self.data_dir / "people").exists("Nik"))

    def test_remove_person_statistics(self):
        stats = StatisticsStore(self.path)
        stats.increment("Nik")
        stats.increment("Luka")
        stats.remove("Nik")
        self.assertEqual(StatisticsStore(self.path).counts, {"Luka": 1})

    def test_rows_order_and_unknown_last(self):
        stats = StatisticsStore(self.path)
        for name, n in (("Luka", 2), ("Unknown", 5), ("Nik", 4)):
            for _ in range(n):
                stats.increment(name, save=False)
        rows = stats.rows(registered=["Nik", "Luka", "Maja"])
        self.assertEqual(rows, [("Nik", 4), ("Luka", 2), ("Maja", 0), ("Unknown", 5)])

    def test_rows_always_contain_unknown(self):
        self.assertEqual(StatisticsStore(self.path).rows(), [("Unknown", 0)])

    def test_corrupted_file_is_backed_up(self):
        self.path.write_text("{broken", encoding="utf-8")
        stats = StatisticsStore(self.path)
        self.assertEqual(stats.counts, {})
        self.assertTrue(self.path.with_name("statistics.json.corrupt").exists())


if __name__ == "__main__":
    unittest.main()
