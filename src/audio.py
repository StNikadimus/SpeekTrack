"""Microphone input via sounddevice (PortAudio).

``Microphone`` delivers fixed-size float32 mono frames at the configured
sample rate. If the device cannot record at 16 kHz directly (common with some
Windows WASAPI and Linux ALSA hardware devices), it records at the device's
native rate and resamples in real time with soxr.
"""

from __future__ import annotations

import queue
from typing import Optional, Union

import numpy as np

try:
    import sounddevice as sd
except OSError as exc:  # PortAudio shared library missing (Linux)
    sd = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class MicrophoneError(RuntimeError):
    pass


def _require_sounddevice() -> None:
    if sd is None:
        raise MicrophoneError(
            f"Audio backend not available ({_IMPORT_ERROR}). On Debian/Ubuntu run: "
            "sudo apt install libportaudio2"
        )


def list_input_devices() -> str:
    """Human readable list of input devices (for troubleshooting)."""
    _require_sounddevice()
    lines = []
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] <= 0:
            continue
        hostapi = sd.query_hostapis(dev["hostapi"])["name"]
        marker = "*" if index == default_in else " "
        lines.append(f"{marker} [{index}] {dev['name']}  ({hostapi}, "
                     f"{int(dev['default_samplerate'])} Hz)")
    if not lines:
        return "No input devices found."
    return "\n".join(lines) + "\n\n* = default input device"


def _check_input_device(device: Optional[Union[int, str]]) -> dict:
    try:
        info = sd.query_devices(device, kind="input")
    except Exception as exc:
        raise MicrophoneError(
            f"No usable microphone found ({exc}). Check that a microphone is "
            "connected and allowed in the OS privacy settings, or set "
            "INPUT_DEVICE in config.py (see `python app.py --list-devices`)."
        ) from exc
    if info["max_input_channels"] < 1:
        raise MicrophoneError(f"Device '{info['name']}' has no input channels.")
    return info


class Microphone:
    """Context manager that streams microphone audio in fixed-size frames.

    >>> with Microphone(16000, 480) as mic:
    ...     frame = mic.read_frame()
    """

    def __init__(self, sample_rate: int, frame_samples: int,
                 device: Optional[Union[int, str]] = None):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self.device_name = ""
        self.overflows = 0
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._stream = None
        self._resampler = None

    # --------------------------------------------------------------- stream
    def _callback(self, indata, frames, time_info, status) -> None:
        if status and status.input_overflow:
            self.overflows += 1
        self._queue.put(indata[:, 0].copy())

    def _open(self, rate: int):
        blocksize = int(round(self.frame_samples * rate / self.sample_rate))
        stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                                blocksize=blocksize, device=self.device,
                                callback=self._callback)
        stream.start()
        return stream

    def __enter__(self) -> "Microphone":
        _require_sounddevice()
        info = _check_input_device(self.device)
        self.device_name = info["name"]
        try:
            self._stream = self._open(self.sample_rate)
        except Exception as direct_error:
            native_rate = int(info["default_samplerate"])
            if native_rate == self.sample_rate:
                raise MicrophoneError(f"Could not open microphone: {direct_error}") from direct_error
            try:
                import soxr
                self._resampler = soxr.ResampleStream(native_rate, self.sample_rate, 1,
                                                      dtype="float32")
                self._stream = self._open(native_rate)
            except Exception as exc:
                raise MicrophoneError(f"Could not open microphone: {exc}") from exc
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    # ----------------------------------------------------------------- read
    def read_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Next frame of exactly ``frame_samples`` samples, or None on timeout."""
        while len(self._buffer) < self.frame_samples:
            try:
                block = self._queue.get(timeout=timeout)
            except queue.Empty:
                return None
            if self._resampler is not None:
                block = self._resampler.resample_chunk(block)
            self._buffer = np.concatenate([self._buffer, block.astype(np.float32)])
        frame = self._buffer[:self.frame_samples]
        self._buffer = self._buffer[self.frame_samples:]
        return frame

    def drain(self) -> None:
        """Drop audio that queued up while we were not reading (e.g. in a prompt)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._buffer = np.zeros(0, dtype=np.float32)
