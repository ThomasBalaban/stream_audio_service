"""
OpenAIRealtimeClient for Stream Audio Service.
Connects to the OpenAI Realtime API via WebSocket, configures server-VAD,
and fires on_transcript(text) for each completed utterance.
"""

import asyncio
import base64
import json
from difflib import SequenceMatcher

import websockets

from config import (
    FORCE_COMMIT_INTERVAL_S,
    OPENAI_API_KEY,
    VAD_PREFIX_PADDING_MS,
    VAD_SILENCE_DURATION_MS,
    VAD_THRESHOLD,
)


class OpenAIRealtimeClient:
    def __init__(self, on_transcript, on_error):
        self.api_key      = OPENAI_API_KEY
        self.on_transcript = on_transcript
        self.on_error      = on_error
        self.ws            = None
        self.loop          = None
        self._closing      = False

        self.url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

        self.audio_accumulated_sec = 0.0
        self.last_transcript       = ""

    async def connect(self):
        self.loop     = asyncio.get_running_loop()
        self._closing = False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta":   "realtime=v1",
        }
        print("🔗 [OpenAI] Connecting to Realtime API …")

        try:
            async with websockets.connect(self.url, additional_headers=headers) as ws:
                self.ws = ws
                print("✅ [OpenAI] Realtime API connected")
                await self._send_session_update()

                async for message in ws:
                    if self._closing:
                        break
                    await self._handle_message(message)

        except asyncio.CancelledError:
            print("🔌 [OpenAI] Connection cancelled")
        except Exception as e:
            if not self._closing:
                print(f"❌ [OpenAI] Connection error: {e}")
                self.on_error(f"Connection failed: {e}")
        finally:
            self.ws = None
            print("[OpenAI] Connection closed")

    async def disconnect(self):
        self._closing = True
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

    async def _send_session_update(self):
        if not self.ws:
            return
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities":              ["text", "audio"],
                "input_audio_format":      "pcm16",
                "input_audio_transcription": {
                    "model":    "whisper-1",
                    "language": "en",
                },
                "turn_detection": {
                    "type":                 "server_vad",
                    "threshold":            VAD_THRESHOLD,
                    "prefix_padding_ms":    VAD_PREFIX_PADDING_MS,
                    "silence_duration_ms":  VAD_SILENCE_DURATION_MS,
                },
            },
        }))
        print(f"📤 [OpenAI] Session configured (VAD {VAD_SILENCE_DURATION_MS}ms, English)")

    def _is_duplicate(self, new_text: str) -> bool:
        if not new_text or len(new_text) < 2:
            return False
        if new_text == self.last_transcript:
            return True
        if self.last_transcript.endswith(new_text):
            return True
        matcher = SequenceMatcher(None, self.last_transcript, new_text)
        match   = matcher.find_longest_match(0, len(self.last_transcript), 0, len(new_text))
        return match.size > len(new_text) * 0.8

    def _filter_transcript(self, text: str) -> str | None:
        if not text:
            return None
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        if len(text) > 0 and (ascii_chars / len(text)) < 0.7:
            return None
        return text

    async def _handle_message(self, message: str):
        try:
            data       = json.loads(message)
            event_type = data.get("type")

            if event_type == "conversation.item.input_audio_transcription.completed":
                text = data.get("transcript", "")
                self.audio_accumulated_sec = 0.0
                if text and text.strip():
                    cleaned = self._filter_transcript(text.strip())
                    if cleaned and not self._is_duplicate(cleaned):
                        self.last_transcript = cleaned
                        self.on_transcript(cleaned)
                    elif cleaned:
                        print(f"♻️  [OpenAI] Deduplicated: {cleaned}")

            elif event_type in (
                "input_audio_buffer.speech_stopped",
                "input_audio_buffer.committed",
            ):
                self.audio_accumulated_sec = 0.0

            elif event_type == "error":
                err     = data.get("error", {})
                err_msg = err.get("message", "Unknown error")
                if "buffer too small" in err_msg or "buffer only has" in err_msg:
                    return
                print(f"❌ [OpenAI] API Error: {err_msg}")
                self.on_error(f"API Error: {err_msg}")

        except Exception as e:
            print(f"[OpenAI] Message parse error: {e}")

    async def send_audio_chunk(self, audio_bytes: bytes):
        if not self.ws or self._closing:
            return
        try:
            encoded = base64.b64encode(audio_bytes).decode("utf-8")
            await self.ws.send(json.dumps({
                "type":  "input_audio_buffer.append",
                "audio": encoded,
            }))

            duration = len(audio_bytes) / 48000.0
            self.audio_accumulated_sec += duration

            if self.audio_accumulated_sec >= FORCE_COMMIT_INTERVAL_S:
                await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                self.audio_accumulated_sec = 0.0

        except Exception as e:
            if not self._closing:
                print(f"[OpenAI] Send failed: {e}")