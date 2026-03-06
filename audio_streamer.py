"""
DesktopAudioStreamer
─────────────────────
Captures desktop audio via sounddevice and streams PCM chunks to the
OpenAI Realtime client. Server-side VAD handles speech segmentation.
Fires volume_callback(level: float) for monitoring.
"""

import asyncio
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from scipy import signal

from config import DESKTOP_AUDIO_DEVICE_ID


class DesktopAudioStreamer:
    def __init__(self, realtime_client, device_id: int = DESKTOP_AUDIO_DEVICE_ID):
        self.client    = realtime_client
        self.device_id = device_id

        # Audio params (resolved at start)
        self.input_rate  = 16000
        self.target_rate = 24000   # OpenAI Realtime expects 24 kHz

        self.queue: queue.Queue = queue.Queue(maxsize=500)
        self.running            = False

        self.volume_callback = None

        # Tuning
        self.gain              = 1.5
        self.remove_dc         = True
        self.db_threshold      = -50     # Only drop truly silent frames
        self.send_interval_s   = 1.2
        self.min_send_samples  = None    # Calculated after device query

        # Threads / loops
        self._process_thread: threading.Thread | None = None
        self._network_thread: threading.Thread | None = None
        self._loop:           asyncio.AbstractEventLoop | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def set_volume_callback(self, cb):
        self.volume_callback = cb

    def start(self):
        self.running         = True
        self._loop           = asyncio.new_event_loop()
        self._network_thread = threading.Thread(
            target=self._network_worker, args=(self._loop,), daemon=True, name="StreamNet"
        )
        self._network_thread.start()
        self._process_thread = threading.Thread(
            target=self._process_worker, daemon=True, name="StreamProc"
        )
        self._process_thread.start()

    def stop(self):
        print("    [Streamer] Stopping …")
        self.running = False

        if self.client and self._loop and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.client.disconnect(), self._loop
                )
                future.result(timeout=2.0)
            except Exception:
                pass

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._network_thread and self._network_thread.is_alive():
            self._network_thread.join(timeout=2.0)
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2.0)

        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break

        print("    [Streamer] Stopped.")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _network_worker(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.client.connect())
        except Exception as e:
            if self.running:
                print(f"[Streamer] Network error: {e}")
        finally:
            try:
                loop.run_forever()
            except Exception:
                pass

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.running:
            return
        float_data = indata.flatten().astype(np.float32) / 32768.0
        rms_val    = np.sqrt(np.mean(float_data ** 2))
        if self.volume_callback:
            self.volume_callback(min(1.0, rms_val * 10))
        try:
            self.queue.put_nowait(indata.copy())
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(indata.copy())
            except Exception:
                pass

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        n = int(len(audio) * target_sr / orig_sr)
        return signal.resample(audio, n)

    def _db(self, audio: np.ndarray) -> float:
        rms = np.sqrt(np.mean(audio ** 2))
        return 20 * np.log10(rms) if rms > 0 else -100

    def _process_worker(self):
        try:
            dev_info         = sd.query_devices(self.device_id, "input")
            self.input_rate  = int(dev_info["default_samplerate"])
            device_name      = dev_info["name"]
        except Exception as e:
            print(f"⚠️  [Streamer] Could not query device {self.device_id}: {e}")
            self.input_rate = 48000
            device_name     = f"Device {self.device_id}"

        self.min_send_samples = int(self.input_rate * self.send_interval_s)

        print(f"🎧 [Streamer] Desktop audio: {device_name}")
        print(f"   Rate: {self.input_rate} Hz → {self.target_rate} Hz | Server VAD enabled")

        time.sleep(2.0)  # Let OpenAI connection establish

        chunk_samples = int(self.input_rate * 0.1)
        retry, max_retry = 0, 5

        while self.running and retry < max_retry:
            try:
                self._run_stream(chunk_samples)
                break
            except Exception as e:
                if not self.running:
                    break
                retry += 1
                print(f"⚠️  [Streamer] Audio error (attempt {retry}/{max_retry}): {e}")
                time.sleep(2.0)

    def _run_stream(self, chunk_samples: int):
        with sd.InputStream(
            device=self.device_id, channels=1, samplerate=self.input_rate,
            callback=self._audio_callback, blocksize=chunk_samples,
            dtype="int16", latency="low",
        ):
            print("✅ [Streamer] Desktop audio stream active")
            audio_buf  = np.array([], dtype=np.float32)
            last_send  = time.time()

            while self.running:
                # Drain the capture queue
                while not self.queue.empty():
                    try:
                        data       = self.queue.get_nowait()
                        chunk_f32  = data.flatten().astype(np.float32) / 32768.0
                        audio_buf  = np.concatenate([audio_buf, chunk_f32])
                    except queue.Empty:
                        break

                now = time.time()
                if (now - last_send >= self.send_interval_s
                        and len(audio_buf) >= self.min_send_samples):

                    to_send   = audio_buf[: self.min_send_samples].copy()
                    audio_buf = audio_buf[self.min_send_samples :]

                    if self.remove_dc:
                        to_send = to_send - np.mean(to_send)
                    to_send = np.clip(to_send * self.gain, -1.0, 1.0)

                    if self._db(to_send) >= self.db_threshold:
                        resampled = self._resample(to_send, self.input_rate, self.target_rate)
                        pcm_bytes = (resampled * 32767).astype(np.int16).tobytes()

                        if self._loop and self._loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self.client.send_audio_chunk(pcm_bytes), self._loop
                            )

                    last_send = now

                # Cap buffer at 5 seconds to prevent drift
                max_buf = int(self.input_rate * 5)
                if len(audio_buf) > max_buf:
                    audio_buf = audio_buf[-max_buf :]

                time.sleep(0.02)