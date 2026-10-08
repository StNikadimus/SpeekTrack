import unittest

import numpy as np

from src import vad as vad_module
from src.vad import VoiceActivityDetector, float_to_pcm16, frame_rms


class VadHelpersTest(unittest.TestCase):
    def test_frame_rms(self):
        self.assertEqual(frame_rms(np.zeros(480, dtype=np.float32)), 0.0)
        self.assertAlmostEqual(frame_rms(np.full(480, 0.5, dtype=np.float32)), 0.5, places=6)

    def test_float_to_pcm16_clips(self):
        pcm = np.frombuffer(float_to_pcm16(np.array([0.0, 1.0, -2.0], dtype=np.float32)), "<i2")
        self.assertEqual(pcm.tolist(), [0, 32767, -32767])


@unittest.skipIf(vad_module.webrtcvad is None, "webrtcvad not installed")
class VoiceActivityDetectorTest(unittest.TestCase):
    def setUp(self):
        self.vad = VoiceActivityDetector(16000, 30, aggressiveness=2, energy_threshold=0.003)

    def test_silence_is_not_speech(self):
        self.assertFalse(self.vad.is_speech(np.zeros(480, dtype=np.float32)))

    def test_quiet_noise_is_gated(self):
        noise = (np.random.default_rng(0).normal(size=480) * 0.001).astype(np.float32)
        self.assertFalse(self.vad.is_speech(noise))

    def test_wrong_frame_size(self):
        with self.assertRaises(ValueError):
            self.vad.is_speech(np.zeros(100, dtype=np.float32))

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            VoiceActivityDetector(16000, 25)
        with self.assertRaises(ValueError):
            VoiceActivityDetector(16000, 30, aggressiveness=5)


if __name__ == "__main__":
    unittest.main()
