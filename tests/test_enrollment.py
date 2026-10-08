import unittest

import numpy as np

from src.enrollment import (LiveSpeakerCheck, progress_bar, record_sample,
                            sample_consistency)
from tests.helpers import FakeEncoder, FakeVAD, near, unit

NIK, LUKA, OTHER = 0.5, 0.3, 0.7


def audio(level, seconds=1.0):
    return np.full(int(16000 * seconds), level, dtype=np.float32)


class LiveSpeakerCheckTest(unittest.TestCase):
    def setUp(self):
        self.luka = unit(2)
        encoder = FakeEncoder({NIK: unit(1), LUKA: self.luka, OTHER: unit(3)})
        self.check = LiveSpeakerCheck(encoder, "Nik", {"Luka": self.luka},
                                      match_threshold=0.7, other_threshold=0.75)

    def test_silence(self):
        status = self.check.check(audio(0.0), voice_detected=False)
        self.assertEqual(status.speaker, "Silence")
        self.assertFalse(status.voice_detected)

    def test_reference_then_match_then_different_voice(self):
        first = self.check.check(audio(NIK), True)
        self.assertIn("building voice reference", first.speaker)
        self.check.check(audio(NIK), True)

        same = self.check.check(audio(NIK), True)
        self.assertEqual(same.speaker, "Nik")
        self.assertTrue(same.matches_target)
        self.assertGreater(same.similarity, 0.9)

        other = self.check.check(audio(OTHER), True)
        self.assertEqual(other.speaker, "Different voice?")
        self.assertFalse(other.matches_target)

    def test_already_registered_voice_is_flagged(self):
        status = self.check.check(audio(LUKA), True)
        self.assertIn("Luka", status.speaker)
        self.assertIn("already registered", status.speaker)
        self.assertEqual(self.check.registered_matches, {"Luka": 1})

    def test_existing_profile_as_initial_reference(self):
        encoder = FakeEncoder({NIK: unit(1)})
        check = LiveSpeakerCheck(encoder, "Nik", {}, 0.7, 0.75, initial_reference=[unit(1)])
        self.assertEqual(check.check(audio(NIK), True).speaker, "Nik")


class FakeMic:
    def __init__(self, signal):
        self.frames = [signal[i:i + 480] for i in range(0, len(signal) - 479, 480)]

    def read_frame(self, timeout=1.0):
        return self.frames.pop(0) if self.frames else None


class RecordSampleTest(unittest.TestCase):
    def test_records_requested_duration_and_reports_status(self):
        signal = np.concatenate([audio(NIK, 3.0), audio(0.0, 1.0)])
        encoder = FakeEncoder({NIK: unit(1)})
        check = LiveSpeakerCheck(encoder, "Nik", {}, 0.7, 0.75)
        lines = []
        sample = record_sample(FakeMic(signal), FakeVAD(), check, duration=4.0,
                               frame_seconds=0.03, check_interval=1.0, check_window=1.6,
                               out=lines.append)
        self.assertAlmostEqual(len(sample.audio) / 16000, 4.0, delta=0.05)
        self.assertAlmostEqual(sample.voiced_seconds, 3.0, delta=0.05)
        self.assertEqual(len(lines), 4)
        self.assertIn("Voice detected: YES", lines[0])
        self.assertIn("Current speaker: Nik", lines[2])
        self.assertIn("Current speaker: Silence", lines[3])
        self.assertEqual(sample.mismatching_windows, 0)
        self.assertEqual(sample.registered_matches, {})

    def test_microphone_failure_raises(self):
        check = LiveSpeakerCheck(FakeEncoder({}), "Nik", {}, 0.7, 0.75)
        with self.assertRaises(RuntimeError):
            record_sample(FakeMic(np.zeros(0, dtype=np.float32)), FakeVAD(), check, 1.0,
                          0.03, 1.0, 1.6, out=lambda _: None)


class HelpersTest(unittest.TestCase):
    def test_sample_consistency(self):
        base = unit(1)
        samples = [near(base, 0.02, 1), near(base, 0.02, 2), unit(50)]
        scores = sample_consistency(samples)
        self.assertGreater(scores[0], 0.8)
        self.assertGreater(scores[1], 0.8)    # not dragged down by the bad sample
        self.assertLess(scores[2], 0.5)       # the odd one out
        self.assertEqual(sample_consistency([base]), [1.0])

    def test_progress_bar(self):
        self.assertEqual(progress_bar(0.0, 4), "[░░░░]")
        self.assertEqual(progress_bar(0.5, 4), "[██░░]")
        self.assertEqual(progress_bar(2.0, 4), "[████]")


if __name__ == "__main__":
    unittest.main()
