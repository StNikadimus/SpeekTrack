import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.speaker_manager import (EMBEDDINGS_FILE, PROFILE_FILE, SpeakerManager,
                                 slugify)
from tests.helpers import near, unit


class SpeakerManagerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.people_dir = Path(self._tmp.name) / "people"
        self.manager = SpeakerManager(self.people_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def samples(self, seed, n=3):
        base = unit(seed)
        return np.stack([near(base, 0.1, seed * 10 + i) for i in range(n)])

    def test_slugify(self):
        self.assertEqual(slugify("Nik"), "nik")
        self.assertEqual(slugify("Anže Novak"), "anze-novak")
        self.assertEqual(slugify("  Čšž!! "), "csz")
        self.assertEqual(slugify("???"), "")

    def test_validate_name(self):
        self.assertIsNone(self.manager.validate_name("Maja"))
        self.assertIsNotNone(self.manager.validate_name(""))
        self.assertIsNotNone(self.manager.validate_name("   "))
        self.assertIsNotNone(self.manager.validate_name("unknown"))  # reserved
        self.assertIsNotNone(self.manager.validate_name("!!!"))
        self.assertIsNotNone(self.manager.validate_name("x" * 100))

    def test_save_creates_expected_files(self):
        self.manager.save("Anže", self.samples(1), [5.0, 6.0, 7.0])
        folder = self.people_dir / "anze"
        self.assertTrue((folder / PROFILE_FILE).exists())
        self.assertTrue((folder / EMBEDDINGS_FILE).exists())
        meta = json.loads((folder / PROFILE_FILE).read_text(encoding="utf-8"))
        self.assertEqual(meta["name"], "Anže")
        self.assertEqual(meta["num_samples"], 3)
        self.assertEqual(meta["embedding_dim"], 256)
        self.assertEqual(np.load(folder / EMBEDDINGS_FILE).shape, (3, 256))

    def test_profiles_survive_restart(self):
        emb = self.samples(2)
        self.manager.save("Nik", emb, [5, 5, 5])
        self.manager.save("Luka", self.samples(3), [5, 5, 5])

        reloaded = SpeakerManager(self.people_dir)  # simulates a program restart
        names = [p.name for p in reloaded.list_profiles()]
        self.assertEqual(names, ["Luka", "Nik"])
        nik = reloaded.get("nik")  # lookup is case-insensitive
        np.testing.assert_allclose(nik.embeddings, emb, rtol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(nik.centroid)), 1.0, places=5)

    def test_append_samples(self):
        self.manager.save("Nik", self.samples(4, n=2), [5, 5])
        created = self.manager.get("Nik").created_at
        self.manager.save("nik", self.samples(5, n=1), [4], append=True)
        profile = self.manager.get("Nik")
        self.assertEqual(profile.num_samples, 3)
        self.assertEqual(profile.name, "Nik")  # original spelling kept
        self.assertEqual(profile.created_at, created)
        self.assertEqual(profile.sample_seconds, [5, 5, 4])

    def test_replace_profile(self):
        self.manager.save("Nik", self.samples(6, n=3), [5, 5, 5])
        self.manager.save("Nik", self.samples(7, n=1), [5])
        self.assertEqual(self.manager.get("Nik").num_samples, 1)

    def test_remove(self):
        self.manager.save("Maja", self.samples(8), [5, 5, 5])
        self.assertTrue(self.manager.exists("Maja"))
        self.assertTrue(self.manager.remove("Maja"))
        self.assertFalse(self.manager.exists("Maja"))
        self.assertFalse((self.people_dir / "maja").exists())
        self.assertFalse(self.manager.remove("Maja"))

    def test_save_rejects_invalid_name(self):
        with self.assertRaises(ValueError):
            self.manager.save("Unknown", self.samples(9), [5, 5, 5])

    def test_broken_profile_is_skipped(self):
        self.manager.save("Nik", self.samples(10), [5, 5, 5])
        broken = self.people_dir / "broken"
        broken.mkdir()
        (broken / PROFILE_FILE).write_text("{not json", encoding="utf-8")
        profiles = self.manager.list_profiles()
        self.assertEqual([p.name for p in profiles], ["Nik"])
        self.assertEqual(len(self.manager.load_errors), 1)

    def test_centroids(self):
        self.manager.save("Nik", self.samples(11), [5, 5, 5])
        centroids = self.manager.centroids()
        self.assertEqual(list(centroids), ["Nik"])
        self.assertEqual(centroids["Nik"].shape, (256,))


if __name__ == "__main__":
    unittest.main()
