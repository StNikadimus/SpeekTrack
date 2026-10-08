"""
Central configuration for Voice Counter.

Every threshold that influences recognition or event counting lives here, so
tuning never requires touching the code. See README.md -> "Configuration" for
a longer explanation of each value.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PEOPLE_DIR = DATA_DIR / "people"            # one sub-folder per registered person
STATISTICS_FILE = DATA_DIR / "statistics.json"

# ---------------------------------------------------------------------------
# Audio input
# ---------------------------------------------------------------------------
# Resemblyzer's encoder works on 16 kHz mono audio. Do not change.
SAMPLE_RATE = 16000

# Length of one analysis frame. webrtcvad only accepts 10, 20 or 30 ms.
FRAME_MS = 30

# Microphone to use. None = operating-system default input device.
# Can also be a device index (int) or part of the device name (str).
# Run `python app.py --list-devices` to see what is available.
INPUT_DEVICE = None

# ---------------------------------------------------------------------------
# Voice activity detection (VAD)
# ---------------------------------------------------------------------------
# webrtcvad aggressiveness, 0 (lenient) .. 3 (strict). Higher values reject
# more background noise but can also cut off quiet speech.
VAD_AGGRESSIVENESS = 2

# Frames whose RMS level is below this value are treated as silence even if
# webrtcvad thinks they contain speech. Float audio is in the range -1..1, so
# 0.003 is roughly -50 dBFS. Raise it in noisy rooms, lower it for quiet mics.
VAD_ENERGY_THRESHOLD = 0.003

# ---------------------------------------------------------------------------
# Speaking events
# ---------------------------------------------------------------------------
# Silence that must pass before the next speech counts as a NEW event.
SPEECH_PAUSE_SECONDS = 1.0

# Voiced audio needed (within a short window) before an event is opened.
# Filters out clicks, door knocks and similar very short sounds.
MIN_SPEECH_START_SECONDS = 0.12

# Events with less voiced audio than this are dropped and NOT counted.
MIN_EVENT_SECONDS = 0.25

# Audio kept from just before speech was detected, so word onsets are not lost.
PRE_SPEECH_PADDING_SECONDS = 0.3

# The speaker is identified as soon as this much voiced audio has been
# collected (or when the event ends, if it is shorter). Longer = more accurate
# but the "X speaking" line appears later.
IDENTIFY_AFTER_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Speaker recognition
# ---------------------------------------------------------------------------
# Minimum cosine similarity between a speech segment and a person's voice
# profile. Below it the segment is reported as "Unknown".
SPEAKER_SIMILARITY_THRESHOLD = 0.75

# The best match must beat the second-best person by at least this much,
# otherwise the segment is "Unknown" (ambiguous between two people).
SPEAKER_MIN_MARGIN = 0.03

# Torch device for the encoder: None = CUDA if available else CPU; or "cpu".
ENCODER_DEVICE = None

# ---------------------------------------------------------------------------
# Enrollment (adding a person)
# ---------------------------------------------------------------------------
# Number of separate recordings per person and the length of each one.
ENROLL_SAMPLES = 3
SAMPLE_DURATION = 8  # seconds

# A recording with less detected speech than this is rejected and repeated.
MIN_ENROLL_SPEECH_SECONDS = 3.0

# Live "Current speaker" check during enrollment: how often, and how much
# recent audio is analysed each time.
ENROLL_CHECK_INTERVAL_SECONDS = 1.0
ENROLL_CHECK_WINDOW_SECONDS = 1.6

# Similarity needed for the live check to say the voice matches the person
# being recorded. Short windows give noisier embeddings, hence a lower value.
ENROLL_LIVE_MATCH_THRESHOLD = 0.65

# After recording, each sample is compared with the others. A sample below
# this similarity is reported as inconsistent and can be re-recorded.
ENROLL_CONSISTENCY_THRESHOLD = 0.75

# If a new profile is this similar to an already registered person, the
# program asks whether it is the same person registered twice. (Profiles of
# the same person typically score > 0.9, different people < 0.75.)
ENROLL_DUPLICATE_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
# Show the similarity score next to each recognised speaker.
SHOW_SIMILARITY = True

# While listening, print the full live statistics table every N events
# (0 = never; statistics are always shown when listening stops).
LIVE_STATS_EVERY_N_EVENTS = 5
