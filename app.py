#!/usr/bin/env python3
"""Voice Counter - real-time speaker recognition and speaking-event counter.

Run:  python app.py              (interactive menu)
      python app.py --list-devices
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import numpy as np

import config
from src.audio import Microphone, MicrophoneError, list_input_devices
from src.counter import StatisticsStore
from src.enrollment import LiveSpeakerCheck, record_sample, sample_consistency
from src.listener import DetectedEvent, ListeningSession
from src.segmenter import SpeechSegmenter
from src.speaker_manager import SpeakerManager, normalize
from src.speaker_recognition import SpeakerEncoder, SpeakerIdentifier, cosine_similarity
from src.vad import VoiceActivityDetector

FRAME_SECONDS = config.FRAME_MS / 1000
FRAME_SAMPLES = config.SAMPLE_RATE * config.FRAME_MS // 1000
WIDTH = 32
MAX_SAMPLE_ATTEMPTS = 3


# --------------------------------------------------------------------- helpers
def header(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(title.center(WIDTH).rstrip())
    print("=" * WIDTH)
    print()


def ask(prompt: str) -> str:
    return input(prompt).strip()


def confirm(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    answer = ask(prompt + suffix).lower()
    if not answer:
        return default
    return answer in ("y", "yes", "d", "da")


class VoiceCounterApp:
    def __init__(self, input_device=config.INPUT_DEVICE):
        self.input_device = input_device
        self.manager = SpeakerManager(config.PEOPLE_DIR)
        self.stats = StatisticsStore(config.STATISTICS_FILE)
        self._encoder = None

    # ----------------------------------------------------------- components
    @property
    def encoder(self):
        """Resemblyzer encoder, loaded once on first use (takes a few seconds)."""
        if self._encoder is None:
            print("Loading voice model (Resemblyzer)...", flush=True)
            try:
                self._encoder = SpeakerEncoder(device=config.ENCODER_DEVICE)
            except (ImportError, OSError) as exc:  # missing package / broken torch install
                raise RuntimeError(
                    f"Could not import Resemblyzer or one of its dependencies ({exc}).\n"
                    "Install them with:\n"
                    "  pip install -r requirements.txt\n"
                    "  pip install --no-deps resemblyzer==0.1.4"
                ) from exc
            print("Voice model ready.")
        return self._encoder

    def make_vad(self) -> VoiceActivityDetector:
        return VoiceActivityDetector(config.SAMPLE_RATE, config.FRAME_MS,
                                     config.VAD_AGGRESSIVENESS, config.VAD_ENERGY_THRESHOLD)

    def microphone(self) -> Microphone:
        return Microphone(config.SAMPLE_RATE, FRAME_SAMPLES, self.input_device)

    # ----------------------------------------------------------------- menu
    def run(self) -> None:
        actions = {
            "1": self.add_person,
            "2": self.start_listening,
            "3": self.show_statistics,
            "4": self.list_people,
            "5": self.remove_person,
            "6": self.reset_statistics,
        }
        while True:
            header("VOICE COUNTER")
            print("1. Add person")
            print("2. Start listening")
            print("3. Show statistics")
            print("4. List people")
            print("5. Remove person")
            print("6. Reset statistics")
            print("7. Exit")
            print()
            try:
                choice = ask("Choose: ")
            except (KeyboardInterrupt, EOFError):
                print()
                choice = "7"
            if choice == "7":
                print("Bye!")
                return
            action = actions.get(choice)
            if action is None:
                print("Please choose a number between 1 and 7.")
                continue
            try:
                action()
            except KeyboardInterrupt:
                print("\nCancelled.")
            except (MicrophoneError, RuntimeError) as exc:
                print(f"\nError: {exc}")
            except EOFError:
                print()
                return

    # ----------------------------------------------------------- add person
    def add_person(self) -> None:
        header("ADD PERSON")
        name = ask("Name: ")
        error = self.manager.validate_name(name)
        if error:
            print(error)
            return

        append = False
        existing = self.manager.get(name) if self.manager.exists(name) else None
        if existing is not None:
            print(f'"{existing.name}" is already registered ({existing.num_samples} samples).')
            choice = ask("[a]dd more samples, [r]eplace the profile or [c]ancel? ").lower()
            if choice.startswith("a"):
                append = True
            elif not choice.startswith("r"):
                print("Cancelled.")
                return
            name = existing.name

        others = {p.name: p.centroid for p in self.manager.list_profiles() if p.name != name}
        encoder = self.encoder
        vad = self.make_vad()

        header(f"Adding person: {name}")
        print("Please speak normally (read something aloud, talk about your day ...).")
        print("Only this person should talk while recording.")
        print(f"{config.ENROLL_SAMPLES} samples x {config.SAMPLE_DURATION} seconds.")

        live_check = LiveSpeakerCheck(
            encoder, name, others,
            match_threshold=config.ENROLL_LIVE_MATCH_THRESHOLD,
            other_threshold=config.SPEAKER_SIMILARITY_THRESHOLD,
            initial_reference=[existing.centroid] if append else None,
        )

        embeddings: List[np.ndarray] = []
        seconds: List[float] = []
        for index in range(config.ENROLL_SAMPLES):
            result = self._record_enrollment_sample(index, live_check, vad)
            if result is None:
                print("Enrollment cancelled - nothing was saved.")
                return
            embeddings.append(result[0])
            seconds.append(result[1])

        if not self._fix_inconsistent_samples(embeddings, seconds, live_check, vad):
            print("Enrollment cancelled - nothing was saved.")
            return

        print()
        print("Voice sample complete.")
        all_embeddings = embeddings + (list(existing.embeddings) if append else [])
        centroid = normalize(np.mean(all_embeddings, axis=0))
        for other_name, other_centroid in others.items():
            sim = cosine_similarity(centroid, other_centroid)
            if sim >= config.ENROLL_DUPLICATE_THRESHOLD:
                print(f'Warning: this voice is almost identical to "{other_name}" ({sim:.2f}). '
                      "Is it the same person? The two would be confused during recognition.")
                if not confirm("Save anyway?"):
                    print("Enrollment cancelled - nothing was saved.")
                    return

        profile = self.manager.save(name, np.stack(embeddings), seconds, append=append)
        print()
        print("Speaker embedding created.")
        if append:
            print(f'Person "{name}" updated ({profile.num_samples} samples in total).')
        else:
            print(f'Person "{name}" successfully registered.')

    def _record_enrollment_sample(self, index: int, live_check: LiveSpeakerCheck,
                                  vad: VoiceActivityDetector):
        """Record one sample (with retries). Returns (embedding, voiced seconds)
        or None if the user gives up."""
        total = config.ENROLL_SAMPLES
        for _ in range(MAX_SAMPLE_ATTEMPTS):
            print()
            ask(f"Press Enter to record sample {index + 1}/{total} ...")
            print(f"Sample {index + 1}/{total} - Recording: {config.SAMPLE_DURATION} seconds")
            print()
            with self.microphone() as mic:
                sample = record_sample(
                    mic, vad, live_check, config.SAMPLE_DURATION, FRAME_SECONDS,
                    config.ENROLL_CHECK_INTERVAL_SECONDS, config.ENROLL_CHECK_WINDOW_SECONDS)

            if sample.voiced_seconds < config.MIN_ENROLL_SPEECH_SECONDS:
                print(f"\nOnly {sample.voiced_seconds:.1f} s of speech detected "
                      f"(need at least {config.MIN_ENROLL_SPEECH_SECONDS:g} s). "
                      "Speak a bit louder / closer to the microphone.")
                continue
            embedding = self.encoder.embed(sample.audio)
            if embedding is None:
                print("\nCould not compute a voice embedding from this sample.")
                continue
            if sample.mismatching_windows > sample.matching_windows:
                if sample.registered_matches:
                    other = max(sample.registered_matches, key=sample.registered_matches.get)
                    print(f'\nWarning: this sounds like "{other}", who is already registered.')
                else:
                    print("\nWarning: most of this sample did not sound like the same person.")
                if not confirm("Keep this sample anyway?"):
                    continue
            print(f"\nSample {index + 1}/{total} OK ({sample.voiced_seconds:.1f} s of speech).")
            return embedding, sample.voiced_seconds
        if confirm(f"Sample {index + 1} failed {MAX_SAMPLE_ATTEMPTS} times. Try again?", True):
            return self._record_enrollment_sample(index, live_check, vad)
        return None

    def _fix_inconsistent_samples(self, embeddings, seconds, live_check, vad) -> bool:
        """Offer to re-record samples that do not resemble the others."""
        while True:
            scores = sample_consistency(embeddings)
            bad = [i for i, s in enumerate(scores) if s < config.ENROLL_CONSISTENCY_THRESHOLD]
            if not bad:
                return True
            print()
            for i in bad:
                print(f"Sample {i + 1} differs from the other samples (similarity {scores[i]:.2f} "
                      f"< {config.ENROLL_CONSISTENCY_THRESHOLD:.2f}).")
            if not confirm("Re-record these samples?", default=True):
                return confirm("Save the profile with the current samples?")
            for i in bad:
                result = self._record_enrollment_sample(i, live_check, vad)
                if result is None:
                    return False
                embeddings[i], seconds[i] = result

    # ------------------------------------------------------------ listening
    def start_listening(self) -> None:
        profiles = self.manager.list_profiles()
        if not profiles:
            print("No people are registered yet - every speaker will be counted as Unknown.")
            if not confirm("Start listening anyway?"):
                return
        encoder = self.encoder
        identifier = SpeakerIdentifier({p.name: p.centroid for p in profiles},
                                       config.SPEAKER_SIMILARITY_THRESHOLD,
                                       config.SPEAKER_MIN_MARGIN)
        segmenter = SpeechSegmenter(
            FRAME_SECONDS, config.SPEECH_PAUSE_SECONDS, config.MIN_SPEECH_START_SECONDS,
            config.MIN_EVENT_SECONDS, config.IDENTIFY_AFTER_SECONDS,
            config.PRE_SPEECH_PADDING_SECONDS)
        vad = self.make_vad()

        header("LISTENING")
        print("Registered: " + (", ".join(p.name for p in profiles) or "-"))
        with self.microphone() as mic:
            print(f"Microphone: {mic.device_name}")
            print("Press Ctrl+C to stop.\n")
            session = ListeningSession(vad, segmenter, encoder, identifier, self.stats,
                                       FRAME_SECONDS, start_time=time.time())
            try:
                while True:
                    frame = mic.read_frame(timeout=2.0)
                    if frame is None:
                        raise MicrophoneError("The microphone stopped delivering audio.")
                    for event in session.process_frame(frame):
                        self._print_event(event, session)
            except KeyboardInterrupt:
                pass
            for event in session.finish():
                self._print_event(event, session)
            overflows = mic.overflows

        print(f"\nStopped listening. {len(session.events)} speaking events in this session.")
        if overflows:
            print(f"(Note: {overflows} audio buffer overflows - the computer was too busy; "
                  "some audio may have been skipped.)")
        self.show_statistics()

    def _print_event(self, event: DetectedEvent, session: ListeningSession) -> None:
        time_str = event.timestamp.strftime("%H:%M:%S")
        match = event.match
        if match.is_known:
            label = f"{match.name} speaking"
            detail = f"similarity {match.similarity:.2f}"
        else:
            label = "Unknown speaker"
            detail = (f"closest: {match.best_candidate} {match.similarity:.2f}"
                      if match.best_candidate else match.reason)
        line = f"[{time_str}] {label}"
        if config.SHOW_SIMILARITY:
            line = f"{line:<32} ({detail}; #{event.count})"
        print(line, flush=True)

        every = config.LIVE_STATS_EVERY_N_EVENTS
        if every and len(session.events) % every == 0:
            self._print_stats_table(indent="    ")

    # ----------------------------------------------------------- statistics
    def _print_stats_table(self, indent: str = "", long: bool = False) -> None:
        rows = self.stats.rows(p.name for p in self.manager.list_profiles())
        width = max(len(name) for name, _ in rows) + 3
        for name, count in rows:
            suffix = f" speaking event{'' if count == 1 else 's'}" if long else ""
            print(f"{indent}{name:<{width}}{count:>4}{suffix}")

    def show_statistics(self) -> None:
        header("STATISTICS")
        self._print_stats_table(long=True)
        print(f"\nTotal: {self.stats.total} speaking events")
        print(f"Counting since: {self.stats.created_at.replace('T', ' ')}")

    def reset_statistics(self) -> None:
        if confirm("Reset ALL statistics? Voice profiles are kept."):
            self.stats.reset()
            print("Statistics reset.")
        else:
            print("Nothing changed.")

    # --------------------------------------------------------------- people
    def list_people(self) -> List[str]:
        profiles = self.manager.list_profiles()
        header("REGISTERED PEOPLE")
        if not profiles:
            print("No people registered yet. Use '1. Add person'.")
        for i, p in enumerate(profiles, 1):
            print(f"{i}. {p.name:<20} {p.num_samples} samples, "
                  f"registered {p.created_at[:10]}")
        for error in self.manager.load_errors:
            print(f"Warning: could not load profile {error}")
        return [p.name for p in profiles]

    def remove_person(self) -> None:
        names = self.list_people()
        if not names:
            return
        choice = ask("\nNumber or name of the person to remove (Enter = cancel): ")
        if not choice:
            return
        name: Optional[str] = None
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            name = names[int(choice) - 1]
        else:
            name = next((n for n in names if n.lower() == choice.lower()), None)
        if name is None:
            print("No such person.")
            return
        if not confirm(f'Remove "{name}" and their voice profile?'):
            print("Nothing changed.")
            return
        self.manager.remove(name)
        print(f'"{name}" removed.')
        if self.stats.get(name) and confirm(f"Also delete {name}'s statistics "
                                            f"({self.stats.get(name)} events)?"):
            self.stats.remove(name)
            print("Statistics deleted.")


def parse_device(value: str):
    return int(value) if value.isdigit() else value


def main(argv=None) -> int:
    # Never crash on consoles that cannot print the progress-bar characters.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description="Real-time speaker recognition and "
                                                 "speaking-event counter.")
    parser.add_argument("--list-devices", action="store_true",
                        help="list available microphones and exit")
    parser.add_argument("--device", type=parse_device, default=config.INPUT_DEVICE,
                        help="input device index or name (overrides INPUT_DEVICE)")
    args = parser.parse_args(argv)

    if args.list_devices:
        try:
            print(list_input_devices())
        except MicrophoneError as exc:
            print(f"Error: {exc}")
            return 1
        return 0

    VoiceCounterApp(input_device=args.device).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
