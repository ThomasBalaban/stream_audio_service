"""
AudioAtmosphereAnalyzer
────────────────────────
Receives raw PCM16 chunks (24kHz, mono) from DesktopAudioStreamer's
atmosphere tap and sends them to GPT-4o Audio for non-speech analysis:
music mood, SFX events, ambience — anything Whisper would miss.

Fires on_atmosphere(atmosphere: dict) for each analyzed chunk.
"""

import base64
import io
import json
import queue
import threading
import time
import wave

from openai import OpenAI

from config import OPENAI_API_KEY

# Only emit an atmosphere event if something actually changed or
# enough time has passed since the last interesting event.
_SILENCE_SKIP_DB   = -55.0   # skip chunks that are effectively silent
_COOLDOWN_SAME_S   = 6.0     # don't re-emit the same mood within N seconds


class AudioAtmosphereAnalyzer:
    def __init__(self, on_atmosphere, sample_rate: int = 24000):
        self.client        = OpenAI(api_key=OPENAI_API_KEY)
        self.on_atmosphere = on_atmosphere
        self.sample_rate   = sample_rate

        self._queue: queue.Queue = queue.Queue(maxsize=20)
        self.running             = False
        self._thread: threading.Thread | None = None

        self._last_mood      = ""
        self._last_mood_time = 0.0
        self._analyze_count  = 0

    # ── Public ───────────────────────────────────────────────────────────────

    def start(self):
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="AtmAnalyzer"
        )
        self._thread.start()
        print("🔊 [AtmAnalyzer] Started (GPT-4o Audio non-speech detection)")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

    def analyze(self, pcm_bytes: bytes):
        """Called by DesktopAudioStreamer's atmosphere tap."""
        try:
            self._queue.put_nowait(pcm_bytes)
        except queue.Full:
            # Drop oldest, keep latest
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(pcm_bytes)
            except Exception:
                pass

    # ── Internal ─────────────────────────────────────────────────────────────

    def _loop(self):
        while self.running:
            try:
                pcm_bytes = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                atmosphere = self._analyze(pcm_bytes)
                if atmosphere:
                    self.on_atmosphere(atmosphere)
            except Exception as e:
                print(f"⚠️  [AtmAnalyzer] Error: {e}")

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        """Wrap raw PCM16 bytes in a WAV container for the API."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)          # int16 = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def _rms_db(self, pcm_bytes: bytes) -> float:
        import struct, math
        samples = struct.unpack(f"{len(pcm_bytes)//2}h", pcm_bytes)
        if not samples:
            return -100.0
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return 20 * math.log10(rms / 32768.0) if rms > 0 else -100.0

    def _analyze(self, pcm_bytes: bytes) -> dict | None:
        # Skip near-silent chunks — nothing interesting to report
        db = self._rms_db(pcm_bytes)
        if db < _SILENCE_SKIP_DB:
            print(f"🔇 [AtmAnalyzer] Skipping silent chunk ({db:.1f} dB)")
            return None

        wav_bytes  = self._pcm_to_wav(pcm_bytes)
        wav_b64    = base64.b64encode(wav_bytes).decode("utf-8")

        prompt = (
            "Listen carefully to this audio clip. Your job is to identify ONLY "
            "non-speech audio content — music, sound effects, ambience, and general "
            "sonic atmosphere. Do NOT transcribe speech; focus entirely on the "
            "background and foreground non-voice sounds.\n\n"
            "Return a JSON object with these fields:\n"
            "{\n"
            '  "music": "<describe the music if present: emotional feel, tempo, '
            'instrumentation, e.g. \'urgent synth pulses with deep bass hits\', '
            '\'gentle acoustic guitar, melancholic\'. null if no music.>",\n'
            '  "sfx": ["<discrete sound event>", "..."],\n'
            '  "ambience": "<continuous background sonic environment, e.g. '
            '\'crowded street with distant sirens\', \'quiet room hum\'. null if none.>",\n'
            '  "mood": "<single evocative phrase capturing the overall audio feel, '
            "e.g. 'tense and urgent', 'playful and light', 'eerie silence', "
            "'chaotic action'. This must always be filled in.>\",\n"
            '  "notable_events": ["<any sudden or prominent audio event worth '
            'flagging, e.g. \'explosion\', \'laser blast\', \'door slam\'. '
            'Empty list if none.>"]\n'
            "}\n\n"
            "Return ONLY valid JSON, no markdown, no explanation."
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data":   wav_b64,
                                "format": "wav",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON if the model wrapped it anyway
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                except Exception:
                    print(f"⚠️  [AtmAnalyzer] Could not parse JSON: {raw[:200]}")
                    return None
            else:
                print(f"⚠️  [AtmAnalyzer] No JSON found in response: {raw[:200]}")
                return None

        # Normalise
        if isinstance(parsed.get("sfx"), str):
            parsed["sfx"] = [parsed["sfx"]] if parsed["sfx"] else []
        if not isinstance(parsed.get("sfx"), list):
            parsed["sfx"] = []
        if isinstance(parsed.get("notable_events"), str):
            parsed["notable_events"] = [parsed["notable_events"]] if parsed["notable_events"] else []
        if not isinstance(parsed.get("notable_events"), list):
            parsed["notable_events"] = []

        # Strip null/empty values
        parsed = {k: v for k, v in parsed.items() if v is not None and v != [] and v != ""}

        mood = parsed.get("mood", "")
        now  = time.time()

        # Suppress identical mood if it just fired recently (no notable events)
        notable = parsed.get("notable_events", [])
        if (mood == self._last_mood
                and now - self._last_mood_time < _COOLDOWN_SAME_S
                and not notable):
            print(f"♻️  [AtmAnalyzer] Suppressing repeat mood: {repr(mood)}")
            return None

        self._analyze_count += 1
        self._last_mood      = mood
        self._last_mood_time = now

        events_str = f" | events: {notable}" if notable else ""
        print(
            f"🔊 [AtmAnalyzer] #{self._analyze_count} "
            f"mood={repr(mood)} | music={repr(str(parsed.get('music',''))[:50])}"
            f"{events_str}"
        )

        return parsed