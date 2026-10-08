import unittest

import numpy as np

from src.segmenter import ENDED, READY, SpeechSegmenter

FRAME = 0.03
FRAME_SAMPLES = 480


def make_segmenter(**overrides):
    params = dict(frame_seconds=FRAME, pause_seconds=1.0, min_start_seconds=0.12,
                  min_event_seconds=0.25, identify_after_seconds=2.0,
                  pre_padding_seconds=0.3)
    params.update(overrides)
    return SpeechSegmenter(**params)


def run(segmenter, pattern, flush=True):
    """pattern: list of (seconds, is_speech). Returns all updates."""
    updates = []
    for seconds, speech in pattern:
        value = 0.5 if speech else 0.0
        for _ in range(int(round(seconds / FRAME))):
            frame = np.full(FRAME_SAMPLES, value, dtype=np.float32)
            updates.extend(segmenter.process(frame, speech))
    if flush:
        updates.extend(segmenter.flush())
    return updates


def ready(updates):
    return [u for u in updates if u.kind == READY]


class SpeakingEventTest(unittest.TestCase):
    def test_single_short_utterance_is_one_event(self):
        # Nik: "Ej, a greva danes ven?"
        updates = run(make_segmenter(), [(0.5, False), (1.5, True), (2.0, False)])
        self.assertEqual(len(ready(updates)), 1)
        self.assertEqual(sum(u.kind == ENDED for u in updates), 1)

    def test_long_speech_with_short_pauses_is_one_event(self):
        # 8 seconds of talking with breathing pauses shorter than 1 s
        pattern = [(0.3, False)]
        for _ in range(4):
            pattern += [(1.6, True), (0.4, False)]
        pattern += [(2.0, False)]
        segmenter = make_segmenter()
        self.assertEqual(len(ready(run(segmenter, pattern))), 1)
        self.assertEqual(segmenter.event_count, 1)

    def test_pause_longer_than_threshold_starts_new_event(self):
        # "Ej." [silence] "Greva?"
        pattern = [(0.3, True), (1.2, False), (0.6, True), (1.5, False)]
        self.assertEqual(len(ready(run(make_segmenter(), pattern))), 2)

    def test_pause_shorter_than_threshold_does_not_split(self):
        pattern = [(0.3, True), (0.7, False), (0.6, True), (1.5, False)]
        self.assertEqual(len(ready(run(make_segmenter(), pattern))), 1)

    def test_pause_threshold_is_configurable(self):
        pattern = [(0.5, True), (0.7, False), (0.5, True), (1.0, False)]
        self.assertEqual(len(ready(run(make_segmenter(pause_seconds=0.5), pattern))), 2)
        self.assertEqual(len(ready(run(make_segmenter(pause_seconds=1.0), pattern))), 1)

    def test_very_short_noise_is_not_counted(self):
        pattern = [(0.06, True), (1.5, False), (0.15, True), (1.5, False)]
        updates = run(make_segmenter(), pattern)
        self.assertEqual(ready(updates), [])
        # the 0.15 s blip opened an event but it ended without being counted
        self.assertTrue(all(not u.counted for u in updates if u.kind == ENDED))

    def test_ready_is_emitted_once_after_identify_seconds(self):
        segmenter = make_segmenter()
        updates = run(segmenter, [(5.0, True)], flush=False)
        self.assertEqual(len(ready(updates)), 1)
        self.assertTrue(segmenter.active)  # event still in progress
        # identification audio: ~2 s of speech plus the pre-speech padding window
        self.assertAlmostEqual(len(ready(updates)[0].audio) / 16000, 2.0, delta=0.4)
        end = segmenter.flush()
        self.assertEqual([u.kind for u in end], [ENDED])
        self.assertTrue(end[0].counted)

    def test_short_event_is_ready_when_it_ends(self):
        segmenter = make_segmenter()
        updates = run(segmenter, [(0.8, True)], flush=False)
        self.assertEqual(ready(updates), [])          # not enough speech yet
        updates = run(segmenter, [(1.1, False)], flush=False)
        self.assertEqual([u.kind for u in updates], [READY, ENDED])
        # trailing silence is trimmed from the identification audio
        self.assertLessEqual(len(updates[0].audio) / 16000, 0.8 + 0.01)

    def test_flush_counts_open_event(self):
        segmenter = make_segmenter()
        run(segmenter, [(0.6, True)], flush=False)
        self.assertEqual(len(ready(segmenter.flush())), 1)
        self.assertEqual(segmenter.flush(), [])

    def test_start_frame_points_at_speech_onset(self):
        updates = run(make_segmenter(), [(1.5, False), (2.5, True), (1.5, False)])
        self.assertAlmostEqual(ready(updates)[0].start_frame * FRAME, 1.5, delta=0.05)

    def test_flickering_onset_still_starts_event(self):
        pattern = [(0.03, True), (0.03, False)] * 10 + [(1.5, False)]
        self.assertEqual(len(ready(run(make_segmenter(), pattern))), 1)

    def test_many_events(self):
        pattern = []
        for _ in range(10):
            pattern += [(1.0, True), (1.3, False)]
        segmenter = make_segmenter()
        self.assertEqual(len(ready(run(segmenter, pattern))), 10)
        self.assertEqual(segmenter.event_count, 10)


if __name__ == "__main__":
    unittest.main()
