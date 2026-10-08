import unittest

from src.speaker_manager import UNKNOWN_NAME
from src.speaker_recognition import SpeakerIdentifier, cosine_similarity
from tests.helpers import near, unit


class UnknownSpeakerTest(unittest.TestCase):
    def setUp(self):
        self.voices = {"Nik": unit(1), "Luka": unit(2), "Anže": unit(3)}
        self.identifier = SpeakerIdentifier(self.voices, threshold=0.75, min_margin=0.03)

    def test_cosine_similarity(self):
        v = unit(5)
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity(v, -v), -1.0, places=5)

    def test_registered_speaker_is_recognised(self):
        match = self.identifier.identify(near(self.voices["Luka"], 0.03, seed=10))
        self.assertTrue(match.is_known)
        self.assertEqual(match.name, "Luka")
        self.assertGreaterEqual(match.similarity, 0.75)

    def test_new_person_is_unknown_not_closest(self):
        # A fourth person who was never registered
        match = self.identifier.identify(unit(99))
        self.assertFalse(match.is_known)
        self.assertEqual(match.name, UNKNOWN_NAME)
        self.assertIsNotNone(match.best_candidate)   # closest is reported...
        self.assertNotEqual(match.name, match.best_candidate)  # ...but not chosen

    def test_below_threshold_is_unknown(self):
        weak = near(self.voices["Nik"], 0.09, seed=11)   # similarity ~0.6
        sim = cosine_similarity(weak, self.voices["Nik"])
        self.assertLess(sim, 0.75)
        match = self.identifier.identify(weak)
        self.assertEqual(match.name, UNKNOWN_NAME)
        self.assertEqual(match.best_candidate, "Nik")
        # ...and the same segment is accepted with a lower threshold
        lenient = SpeakerIdentifier(self.voices, threshold=sim - 0.01)
        self.assertEqual(lenient.identify(weak).name, "Nik")

    def test_ambiguous_between_two_people_is_unknown(self):
        a = unit(20)
        b = near(a, 0.01, seed=21)  # two almost identical voices
        identifier = SpeakerIdentifier({"Maja": a, "Ana": b}, threshold=0.5, min_margin=0.05)
        match = identifier.identify(near(a, 0.01, seed=22))
        self.assertEqual(match.name, UNKNOWN_NAME)
        self.assertIn("too close", match.reason)

    def test_no_registered_people(self):
        match = SpeakerIdentifier({}, threshold=0.75).identify(unit(1))
        self.assertEqual(match.name, UNKNOWN_NAME)
        self.assertIsNone(match.best_candidate)

    def test_missing_embedding_is_unknown(self):
        self.assertEqual(self.identifier.identify(None).name, UNKNOWN_NAME)

    def test_scores_for_every_person(self):
        match = self.identifier.identify(unit(1))
        self.assertEqual(set(match.scores), {"Nik", "Luka", "Anže"})


if __name__ == "__main__":
    unittest.main()
