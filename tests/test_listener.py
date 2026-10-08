"""End-to-end pipeline test with fake audio, VAD and encoder."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.counter import StatisticsStore
from src.listener import ListeningSession
from src.segmenter import SpeechSegmenter
from src.speaker_recognition import SpeakerIdentifier
from tests.helpers import FakeEncoder, FakeVAD, unit

FRAME = 0.03
FRAME_SAMPLES = 480
NIK, LUKA, STRANGER = 0.5, 0.3, 0.7   # "audio level" that identifies each voice


class ListeningPipelineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.stats_path = Path(self._tmp.name) / "statistics.json"
        nik, luka = unit(1), unit(2)
        self.encoder = FakeEncoder({NIK: nik, LUKA: luka, STRANGER: unit(3)})
        self.session = ListeningSession(
            vad=FakeVAD(),
            segmenter=SpeechSegmenter(FRAME, 1.0, 0.12, 0.25, 2.0, 0.3),
            encoder=self.encoder,
            identifier=SpeakerIdentifier({"Nik": nik, "Luka": luka}, 0.75, 0.03),
            stats=StatisticsStore(self.stats_path),
            frame_seconds=FRAME,
            start_time=1_700_000_000.0,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def feed(self, pattern):
        events = []
        for seconds, level in pattern:
            for _ in range(int(round(seconds / FRAME))):
                frame = np.full(FRAME_SAMPLES, level, dtype=np.float32)
                events.extend(self.session.process_frame(frame))
        return events

    def test_conversation(self):
        events = self.feed([
            (0.5, 0.0),
            (3.0, NIK),                    # Nik talks (one event)
            (1.5, 0.0),
            (1.0, LUKA), (0.5, 0.0),       # Luka, short pause...
            (1.0, LUKA),                   # ...continues -> same event
            (1.5, 0.0),
            (2.5, STRANGER),               # unregistered person
            (1.5, 0.0),
            (0.4, NIK),                    # "Ej."
            (1.2, 0.0),
            (0.6, NIK),                    # "Greva?" -> separate event
            (1.5, 0.0),
        ])
        events += self.session.finish()

        self.assertEqual([e.match.name for e in events],
                         ["Nik", "Luka", "Unknown", "Nik", "Nik"])
        self.assertEqual(self.session.stats.counts, {"Nik": 3, "Luka": 1, "Unknown": 1})
        # the model runs once per event, not once per frame
        self.assertEqual(self.encoder.calls, 5)
        # running count per person is reported with each event
        self.assertEqual([e.count for e in events], [1, 1, 1, 2, 3])

    def test_unknown_is_never_auto_registered(self):
        self.feed([(2.5, STRANGER), (1.5, 0.0), (2.5, STRANGER), (1.5, 0.0)])
        self.assertEqual(self.session.stats.counts, {"Unknown": 2})
        self.assertEqual(set(self.session.identifier.centroids), {"Nik", "Luka"})

    def test_counts_are_persisted_immediately(self):
        self.feed([(2.5, NIK), (1.5, 0.0)])
        self.assertEqual(StatisticsStore(self.stats_path).counts, {"Nik": 1})

    def test_event_timestamp_is_speech_start(self):
        events = self.feed([(4.0, 0.0), (2.5, LUKA), (1.5, 0.0)])
        offset = events[0].timestamp.timestamp() - 1_700_000_000.0
        self.assertAlmostEqual(offset, 4.0, delta=0.1)

    def test_utterance_in_progress_is_counted_on_stop(self):
        self.feed([(1.0, LUKA)])          # still talking when the user stops
        events = self.session.finish()
        self.assertEqual([e.match.name for e in events], ["Luka"])


if __name__ == "__main__":
    unittest.main()
