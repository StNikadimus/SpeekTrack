# Voice Counter

Small, local, real-time **speaker recognition + speaking-event counter**.

```
MICROPHONE → VOICE ACTIVITY DETECTION → SPEAKER EMBEDDING
           → COMPARE WITH REGISTERED PEOPLE → IDENTIFY SPEAKER → COUNT SPEAKING EVENT
```

## Overview

Voice Counter listens to your microphone and:

* detects when somebody starts talking (voice activity detection),
* recognises **which registered person** is speaking, using voice embeddings
  from [Resemblyzer](https://github.com/resemble-ai/Resemblyzer),
* adds **one** to that person's counter for every new *speaking event*
  (an utterance separated from the previous one by a pause),
* reports the voice as **Unknown** when it is not similar enough to any
  registered person — it never forces the closest match,
* lets you register new people in about half a minute.

Everything runs locally. No speech-to-text, no cloud APIs, no LLMs; the audio
is never stored, only voice embeddings (256 numbers per sample) and counts.

```
================================
          LISTENING
================================

Registered: Luka, Maja, Nik
Microphone: MacBook Pro Microphone
Press Ctrl+C to stop.

[14:32:04] Nik speaking          (similarity 0.80; #1)
[14:32:08] Luka speaking         (similarity 0.89; #1)
[14:32:11] Unknown speaker       (closest: Maja 0.58; #1)
[14:32:16] Maja speaking         (similarity 0.85; #1)
[14:32:21] Nik speaking          (similarity 0.82; #2)
```

## Requirements

| What | Details |
|------|---------|
| Python | **3.10 – 3.13** (developed and tested with 3.13). 64-bit. |
| Microphone | Any microphone the operating system can use (built-in, USB, headset). |
| Model | Resemblyzer's pretrained voice encoder. **It is bundled inside the `resemblyzer` pip package** (`pretrained.pt`, ~17 MB) — nothing is downloaded at runtime. |
| OS | Windows 10/11, macOS, Linux. |
| Linux only | PortAudio library: `sudo apt install libportaudio2` (Debian/Ubuntu), `sudo dnf install portaudio` (Fedora). Windows and macOS wheels of `sounddevice` already include it. |
| Disk | ~1–1.5 GB for the virtual environment on Windows/macOS. On Linux the default PyTorch wheel pulls in CUDA libraries (~6 GB in total); see *Installation* for the CPU-only alternative. |
| CPU | Any recent CPU. One embedding takes ~50 ms on a laptop CPU; no GPU needed. |

## Installation

```bash
git clone <this repository> voice-counter
cd voice-counter
```

Create and activate a virtual environment:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

(cmd.exe: `.venv\Scripts\activate.bat`. If PowerShell refuses to run the
script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies — **two commands**:

```bash
pip install -r requirements.txt
pip install --no-deps resemblyzer==0.1.4
```

Why the second command uses `--no-deps`: Resemblyzer 0.1.4 (the latest
release; its GitHub `master` is identical) declares a dependency on the
`webrtcvad` package, which only exists as C source code and fails to install
without a C compiler (typically on Windows). `requirements.txt` instead
installs `webrtcvad-wheels`, a prebuilt drop-in that provides the same
`webrtcvad` module, plus all of Resemblyzer's other dependencies. (`pip check`
will therefore mention that `webrtcvad` and `typing` are "missing" — that is
expected and harmless; `typing` is part of Python itself.)

**Optional (Linux): CPU-only PyTorch** — saves several GB of disk. Run this
*before* `pip install -r requirements.txt`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Check that the microphone is visible:

```bash
python app.py --list-devices
```

## Usage

```bash
python app.py
```

```
================================
         VOICE COUNTER
================================

1. Add person
2. Start listening
3. Show statistics
4. List people
5. Remove person
6. Reset statistics
7. Exit

Choose:
```

Command-line options:

| Option | Meaning |
|--------|---------|
| `--list-devices` | List microphones and exit. |
| `--device N` / `--device "USB"` | Use input device index `N` or the first device whose name contains the text (overrides `INPUT_DEVICE` in `config.py`). |

The voice model loads the first time you add a person or start listening
(a few seconds) and then stays loaded.

### Adding people

1. Choose **1. Add person** and type a name (e.g. `Nik`).
2. The program records **3 samples × 8 seconds** (configurable). Press Enter
   before each sample and **talk normally** — read a paragraph aloud, describe
   your day. Only the person being registered should talk.
3. While recording, a line is printed every second:

   ```
   [█████░░░░░░░░░░░░░░░]  2.0/8s  Voice detected: YES  Current speaker: Nik (building voice reference...)
   [███████░░░░░░░░░░░░░]  3.0/8s  Voice detected: YES  Current speaker: Nik (0.86)
   [██████████░░░░░░░░░░]  4.0/8s  Voice detected: NO   Current speaker: Silence
   [█████████████░░░░░░░]  5.0/8s  Voice detected: YES  Current speaker: Different voice? (0.52)
   [███████████████░░░░░]  6.0/8s  Voice detected: YES  Current speaker: Luka (already registered!)
   ```

   * `Nik (0.86)` — the voice matches the reference built from the first
     seconds of this enrollment (number = similarity).
   * `Different voice?` — somebody else (or noise) was heard.
   * `Luka (already registered!)` — the voice sounds like an existing person.
   * `Silence` — no speech detected.

4. Quality checks after each sample and at the end:
   * a sample with less than 3 s of speech is repeated automatically,
   * a sample that mostly did not sound like the person (or sounded like
     somebody already registered) is shown with a warning and you can retry,
   * samples that do not resemble each other can be re-recorded,
   * if the whole profile is almost identical to an existing person
     (same person registered twice?) you are asked before saving.
5. Done:

   ```
   Voice sample complete.

   Speaker embedding created.
   Person "Nik" successfully registered.
   ```

Adding a name that already exists lets you **add more samples** (improves
recognition, e.g. samples recorded in another room / with another mic) or
**replace** the profile.

### Listening

Choose **2. Start listening**. Every speaking event prints one line with the
time the speech started. With `SHOW_SIMILARITY = True` it also shows the
similarity and the person's running count (`#3` = third event). Every
`LIVE_STATS_EVERY_N_EVENTS` events a live statistics table is printed.

Press **Ctrl+C** to stop. An utterance that is still in progress is counted,
the statistics are shown, and you return to the menu. Counts are saved to disk
after every event, so nothing is lost even if the program is killed.

### Speaking events

A *speaking event* is one continuous utterance. The audio is split into 30 ms
frames, and each frame is classified as speech/no speech by
[webrtcvad](https://github.com/wiseman/py-webrtcvad) plus an energy gate.

* An event **starts** when at least `MIN_SPEECH_START_SECONDS` of speech frames
  appear within a short window (filters clicks and knocks).
* Short pauses inside the utterance (breathing, between words) are ignored.
* The event **ends** after `SPEECH_PAUSE_SECONDS` (default **1.0 s**) of
  continuous silence. Speech after that is a **new** event.
* Events with less than `MIN_EVENT_SECONDS` of speech are ignored as noise.
* The speaker is identified **once per event**, as soon as
  `IDENTIFY_AFTER_SECONDS` (2 s) of speech is available, or when the event
  ends if it is shorter. The counter is incremented exactly then — once per
  event, never per frame.

Examples (with the default 1.0 s pause):

| Audio | Events |
|-------|--------|
| Nik: "Ej, a greva danes ven?" | 1 |
| Nik talks for 8 s, only short breaths in between | 1 |
| Nik: "Ej." · *1.2 s silence* · Nik: "Greva?" | 2 |
| Nik: "Ej." · *0.5 s silence* · Nik: "Greva?" | 1 |

### Unknown speakers

Each registered person is represented by the average (centroid) of their
sample embeddings. A new speech segment is embedded and compared with every
person using **cosine similarity** (1.0 = identical voice).

The segment is attributed to the best-matching person **only if both** hold:

1. similarity ≥ `SPEAKER_SIMILARITY_THRESHOLD` (default **0.75**), and
2. it beats the second-best person by at least `SPEAKER_MIN_MARGIN` (0.03).

Otherwise the result is `Unknown speaker`, counted under **Unknown**. The
closest person is shown for information (`closest: Maja 0.58`) but is *not*
counted. Unknown voices are never registered automatically — use
*Add person* if you want to track them.

Reference numbers measured with this pipeline on LibriSpeech recordings
(10 speakers, 2 s segments vs. a 3-sample profile): same person ≈ 0.83 on
average (94 % ≥ 0.75), different people ≤ 0.70. Live microphone audio in a
real room is noisier, so check the similarities printed while listening and
tune the threshold (see *Troubleshooting*).

## Configuration

All tunable values live in [`config.py`](config.py).

| Setting | Default | Meaning |
|---------|---------|---------|
| `SPEECH_PAUSE_SECONDS` | `1.0` | Silence that separates two speaking events. Larger = fewer, longer events. |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.75` | Minimum similarity to accept a registered person. Higher = fewer false matches, more Unknown. Typical range 0.65–0.85. |
| `SPEAKER_MIN_MARGIN` | `0.03` | Best person must beat the runner-up by this much, else Unknown. |
| `SAMPLE_DURATION` | `8` | Seconds per enrollment sample. |
| `ENROLL_SAMPLES` | `3` | Number of enrollment samples per person. |
| `MIN_ENROLL_SPEECH_SECONDS` | `3.0` | A sample with less detected speech is repeated. |
| `IDENTIFY_AFTER_SECONDS` | `2.0` | Speech collected before identifying the speaker. Longer = more accurate, later output. |
| `MIN_SPEECH_START_SECONDS` | `0.12` | Speech needed to open an event. |
| `MIN_EVENT_SECONDS` | `0.25` | Shorter events are not counted. |
| `PRE_SPEECH_PADDING_SECONDS` | `0.3` | Audio kept from before the detected onset. |
| `VAD_AGGRESSIVENESS` | `2` | webrtcvad mode 0–3; higher rejects more noise. |
| `VAD_ENERGY_THRESHOLD` | `0.003` | Frames quieter than this RMS are silence (≈ −50 dBFS). |
| `INPUT_DEVICE` | `None` | Microphone (index or name part); `None` = system default. |
| `ENCODER_DEVICE` | `None` | `None` = CUDA if available, else CPU; or `"cpu"`. |
| `ENROLL_LIVE_MATCH_THRESHOLD` | `0.65` | Similarity for the live "Current speaker" check during enrollment (short windows → lower value). |
| `ENROLL_CONSISTENCY_THRESHOLD` | `0.75` | Samples less similar to each other than this can be re-recorded. |
| `ENROLL_DUPLICATE_THRESHOLD` | `0.85` | Above this, a new profile is suspected to be an already-registered person. |
| `ENROLL_CHECK_INTERVAL_SECONDS` / `ENROLL_CHECK_WINDOW_SECONDS` | `1.0` / `1.6` | How often / how much audio the live enrollment check analyses. |
| `SHOW_SIMILARITY` | `True` | Print similarity and running count with each event. |
| `LIVE_STATS_EVERY_N_EVENTS` | `5` | Print a live statistics table every N events (0 = off). |
| `SAMPLE_RATE`, `FRAME_MS` | `16000`, `30` | Fixed by Resemblyzer / webrtcvad — do not change. |

## Data and persistence

Voice profiles and statistics are stored separately, so statistics can be
reset without touching the voice profiles:

```
data/
├── people/
│   ├── nik/
│   │   ├── profile.json     # name, dates, number of samples, seconds of speech
│   │   └── embeddings.npy   # one 256-d Resemblyzer embedding per sample
│   └── anze/                # folder = name without accents/spaces ("Anže")
│       └── ...
└── statistics.json          # {"counts": {"Nik": 37, "Unknown": 4, ...}}
```

* Profiles are loaded from disk on every start — no re-recording needed.
* **6. Reset statistics** clears `statistics.json` only.
* **5. Remove person** deletes the person's folder and asks whether to delete
  their statistics too.
* Writes are atomic (temp file + rename), so a crash cannot leave half a file.
* `data/` is in `.gitignore` — voice data is personal and should not be
  committed.

## Project structure

```
voice-counter/
├── app.py                      # CLI menu, wiring of all components
├── config.py                   # all thresholds and paths
├── requirements.txt
├── src/
│   ├── audio.py                # microphone stream (sounddevice), device listing, resampling fallback
│   ├── vad.py                  # webrtcvad + energy gate, per 30 ms frame
│   ├── segmenter.py            # frames → speaking events (pause logic)
│   ├── speaker_recognition.py  # Resemblyzer encoder wrapper, open-set matching (Unknown)
│   ├── speaker_manager.py      # voice profiles on disk
│   ├── enrollment.py           # recording samples + live "Current speaker" check
│   ├── listener.py             # real-time pipeline: VAD → events → speaker → counter
│   └── counter.py              # statistics (separate JSON file)
├── data/
└── tests/
```

## Running the tests

```bash
python -m unittest -v
```

The tests use only the standard library (`unittest`; `pytest` also works) and
need neither a microphone nor the neural network. They cover profile
management, saving/loading embeddings, the statistics counter,
speaking-event segmentation, Unknown-speaker decisions, the live enrollment
check and the full listening pipeline with simulated audio.

## Troubleshooting

**Microphone not found / "No usable microphone found"**
* Run `python app.py --list-devices`. The default input is marked with `*`.
* Choose another device: `python app.py --device 2` or set `INPUT_DEVICE` in
  `config.py`.
* Windows: *Settings → Privacy & security → Microphone* → allow desktop apps.
  macOS: *System Settings → Privacy & Security → Microphone* → allow your
  terminal app (Terminal / iTerm / VS Code), then restart it.
* Linux: `OSError: PortAudio library not found` → `sudo apt install libportaudio2`.
  Make sure your user can record (`arecord -l`; PipeWire/PulseAudio users can
  pick the device `pulse` or `pipewire`).

**Low speaker recognition accuracy**
* Look at the similarities printed while listening. If correct speakers are
  often just below the threshold (e.g. 0.70–0.74) → lower
  `SPEAKER_SIMILARITY_THRESHOLD` to ~0.70. If wrong people are recognised →
  raise it to ~0.80.
* Record the enrollment with the **same microphone and room** that you use for
  listening. Use *Add person → add more samples* to add samples from other
  conditions.
* Short utterances ("Ja.", "Ej.") contain little voice information and often
  end up as Unknown; that is expected (see *Limitations*).
* Keep a distance of ~0.5–1 m to the microphone; avoid clipping.

**Resemblyzer / model issues**
* `Could not import Resemblyzer` → run both install commands from
  *Installation* inside the activated virtual environment.
* `pip install resemblyzer` fails building `webrtcvad` → you forgot `--no-deps`.
* There is no model download: the model file ships with the package. If you
  see a "Couldn't find the voice encoder pretrained model" error, reinstall:
  `pip install --force-reinstall --no-deps resemblyzer==0.1.4`.
* A `DeprecationWarning` about `scipy.ndimage.morphology` is silenced by the
  program; keep `scipy<2` as in `requirements.txt`.
* Windows `OSError: [WinError 126] ... torch ... .dll` → install the
  [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)
  (x64) and use 64-bit Python.

**Windows audio issues**
* "Invalid sample rate" errors are handled automatically: the program records
  at the device's native rate and resamples to 16 kHz.
* If the default device is a WASAPI/WDM-KS endpoint that refuses to open, pick
  the same microphone under the **MME** host API from `--list-devices`.
* Disable "audio enhancements" / noise suppression of headsets if recognition is
  poor; they distort the voice.
* Close other programs that use the microphone in exclusive mode (some
  conferencing apps).

**Background noise**
* Noise opens false events (counted as Unknown) → raise
  `VAD_ENERGY_THRESHOLD` (e.g. `0.01`) and/or `VAD_AGGRESSIVENESS` to `3`.
* Quiet speakers are missed → lower `VAD_ENERGY_THRESHOLD` (e.g. `0.001`).
* Fans/music close to the mic reduce similarity → move the microphone closer
  to the speakers.

**"audio buffer overflows" note after listening**
The computer was too busy to read audio in time. Close heavy programs; on
slow machines set `ENCODER_DEVICE = "cpu"` and keep `IDENTIFY_AFTER_SECONDS`
at 2 s.

## Limitations

* **Overlapping speech** (two people talking at once) produces a mixed
  embedding — usually Unknown or the louder speaker.
* **Fast turn-taking without a pause** (Luka answers less than
  `SPEECH_PAUSE_SECONDS` after Nik stops) is merged into one event attributed
  to the first speaker. Lowering `SPEECH_PAUSE_SECONDS` (e.g. 0.6) helps.
* **Similar voices** (siblings, same-sex speakers with similar timbre) can be
  confused or end up Unknown because of the margin rule.
* **Noise and reverberation** lower similarity and increase Unknown counts;
  loud noise can create false events.
* **Microphone quality** and distance matter a lot; enrollment and listening
  should use the same setup.
* **Very short utterances** (< ~1 s) are harder to identify; they are counted,
  but more often as Unknown.
* Resemblyzer is a 2019-era model (GE2E, trained mostly on English speech);
  it works for any language, but newer embedding models are more accurate.
* One microphone, one room — no speaker localisation or diarization of
  recorded files.
