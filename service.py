"""
StreamAudioService
──────────────────
• Streams desktop audio to OpenAI Realtime API (Whisper VAD transcription)
• Optionally enriches transcripts with GPT-4o (speaker IDs, tone, etc.)
• Subscribes to Hub vision_context events so the enricher has visual context
• Broadcasts results to:
    - WebSocket clients on port 8017
    - The central Hub (audio_context events)
"""

import asyncio
import threading
import time
import uuid

import socketio

from audio_streamer import DesktopAudioStreamer
from config import (
    DESKTOP_AUDIO_DEVICE_ID,
    ENABLE_ENRICHMENT,
    HUB_URL,
    OPENAI_API_KEY,
    SERVICE_NAME,
)
from openai_realtime_client import OpenAIRealtimeClient
from transcript_enricher import TranscriptEnricher
from websocket_server import WebSocketServer


class StreamAudioService:
    def __init__(self):
        print(f"🔊 [{SERVICE_NAME}] Initializing …")

        self._shutting_down = False
        self._shutdown_lock = threading.Lock()

        if not OPENAI_API_KEY:
            raise RuntimeError(
                f"[{SERVICE_NAME}] OPENAI_API_KEY not found in api_keys.py"
            )

        # ── WebSocket broadcast server ────────────────────────────────────────
        self.ws_server = WebSocketServer()

        # ── Socket.IO hub client ──────────────────────────────────────────────
        self.sio      = socketio.AsyncClient(reconnection=True, reconnection_delay=5)
        self.hub_loop: asyncio.AbstractEventLoop | None = None
        self._register_hub_events()

        # ── OpenAI Realtime client ────────────────────────────────────────────
        self.openai_client = OpenAIRealtimeClient(
            on_transcript = self._on_whisper_transcript,
            on_error      = self._on_openai_error,
        )

        # ── Desktop audio streamer ────────────────────────────────────────────
        self.streamer = DesktopAudioStreamer(
            realtime_client = self.openai_client,
            device_id       = DESKTOP_AUDIO_DEVICE_ID,
        )
        self.streamer.set_volume_callback(self._on_volume)

        # ── Transcript enricher (optional) ────────────────────────────────────
        self.enricher: TranscriptEnricher | None = None
        if ENABLE_ENRICHMENT:
            self.enricher = TranscriptEnricher(
                on_enriched_transcript=self._on_enriched_transcript,
            )

    # ── Public ───────────────────────────────────────────────────────────────

    def run(self):
        # Hub event-loop
        self.hub_loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._hub_thread, args=(self.hub_loop,), daemon=True, name="StreamHub"
        ).start()

        # WebSocket
        self.ws_server.start()

        # Enricher
        if self.enricher:
            self.enricher.start()

        # Desktop audio streamer (starts OpenAI connection internally)
        self.streamer.start()

        print(f"✅ [{SERVICE_NAME}] All components running. Press Ctrl-C to stop.")

        try:
            while not self._shutting_down:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True

        print(f"🛑 [{SERVICE_NAME}] Shutting down …")

        try:
            self.streamer.stop()
        except Exception:
            pass

        if self.enricher:
            try:
                self.enricher.stop()
            except Exception:
                pass

        try:
            self.ws_server.stop()
        except Exception:
            pass

        if self.hub_loop:
            self.hub_loop.call_soon_threadsafe(self.hub_loop.stop)

        print(f"🛑 [{SERVICE_NAME}] Stopped.")

    # ── Hub: connection + events ──────────────────────────────────────────────

    def _register_hub_events(self):
        @self.sio.on("vision_context")
        async def on_vision_context(data):
            """Feed visual context into the enricher."""
            ctx = data.get("context", "")
            if ctx and self.enricher:
                self.enricher.update_visual_context(ctx)

        @self.sio.on("text_update")
        async def on_text_update(data):
            ctx = data.get("content", "")
            if ctx and self.enricher:
                self.enricher.update_visual_context(ctx)

    def _hub_thread(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.create_task(self._hub_connection_loop())
        loop.run_forever()

    async def _hub_connection_loop(self):
        while not self._shutting_down:
            if not self.sio.connected:
                try:
                    await self.sio.connect(HUB_URL)
                    print(f"✅ [{SERVICE_NAME}] Hub connected: {HUB_URL}")
                except Exception as e:
                    print(f"⚠️  [{SERVICE_NAME}] Hub connect failed: {e}. Retrying …")
                    await asyncio.sleep(5)
            await asyncio.sleep(2)

    def _emit_to_hub(self, event: str, data: dict):
        if self.sio.connected and self.hub_loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.sio.emit(event, data), self.hub_loop
                )
            except Exception as e:
                print(f"❌ [{SERVICE_NAME}] Hub emit error: {e}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_volume(self, level: float):
        self.ws_server.broadcast({"type": "volume", "source": "desktop", "level": level})

    def _on_whisper_transcript(self, raw_text: str):
        """Called by OpenAI Realtime on completed utterance."""
        tid = str(uuid.uuid4())

        # Broadcast raw transcript immediately
        self.ws_server.broadcast({
            "type":      "transcript_raw",
            "source":    "desktop",
            "text":      raw_text,
            "id":        tid,
            "timestamp": time.time(),
        })

        if self.enricher:
            # Queue for GPT-4o enrichment; _on_enriched_transcript fires next
            self.enricher.enrich(raw_text, transcript_id=tid)
        else:
            # No enricher: publish directly
            self._publish_transcript(raw_text, tid, enriched=False)

    def _on_enriched_transcript(self, enriched_text: str, transcript_id: str | None = None):
        """Called by TranscriptEnricher when GPT-4o enrichment is done."""
        speaker = "Unknown"
        import re
        match = re.search(
            r"\[\d+:\d+\]\s*(?:\[.*?\]\s*)?([^:(]+?)(?:\s*\([^)]+\))?:", enriched_text
        )
        if match:
            speaker = match.group(1).strip()

        # Local WebSocket
        self.ws_server.broadcast({
            "type":      "transcript_enriched",
            "source":    "desktop",
            "speaker":   speaker,
            "text":      enriched_text,
            "enriched":  True,
            "id":        transcript_id,
            "timestamp": time.time(),
        })

        # Hub
        self._emit_to_hub("audio_context", {
            "context":    enriched_text,
            "is_partial": False,
            "metadata":   {
                "source":  "desktop",
                "speaker": speaker,
                "id":      transcript_id,
            },
        })

        print(f"🔊 [{SERVICE_NAME}] {speaker}: {enriched_text[:100]}")

    def _publish_transcript(self, text: str, tid: str, enriched: bool):
        self.ws_server.broadcast({
            "type":      "transcript",
            "source":    "desktop",
            "text":      text,
            "enriched":  enriched,
            "id":        tid,
            "timestamp": time.time(),
        })
        self._emit_to_hub("audio_context", {
            "context":    text,
            "is_partial": False,
            "metadata":   {"source": "desktop", "id": tid},
        })
        print(f"🔊 [{SERVICE_NAME}] → {text[:100]}")

    def _on_openai_error(self, msg: str):
        print(f"❌ [{SERVICE_NAME}] OpenAI error: {msg}")
        self.ws_server.broadcast({"type": "error", "source": "openai", "message": msg})