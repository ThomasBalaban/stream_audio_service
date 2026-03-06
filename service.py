"""
StreamAudioService — full verbose logging
"""
import asyncio
import re
import threading
import time
import traceback
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


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [{SERVICE_NAME}] {msg}", flush=True)


class StreamAudioService:
    def __init__(self):
        log("🔊 Initializing …")
        self._shutting_down      = False
        self._shutdown_lock      = threading.Lock()
        self._hub_emit_count     = 0
        self._ws_broadcast_count = 0
        self._whisper_count      = 0
        self._enriched_count     = 0
        self._vision_ctx_count   = 0
        self._atmosphere_count   = 0

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not found in api_keys.py")
        log("✅ OPENAI_API_KEY present")

        # ── WebSocket server ──────────────────────────────────────────────────
        log("Creating WebSocket broadcast server …")
        self.ws_server = WebSocketServer()

        # ── Socket.IO hub client ──────────────────────────────────────────────
        self.sio      = socketio.AsyncClient(reconnection=True, reconnection_delay=5)
        self.hub_loop: asyncio.AbstractEventLoop | None = None
        self._register_hub_events()

        # ── OpenAI Realtime client ────────────────────────────────────────────
        log("Initializing OpenAIRealtimeClient …")
        try:
            self.openai_client = OpenAIRealtimeClient(
                on_transcript = self._on_whisper_transcript,
                on_error      = self._on_openai_error,
            )
            log("✅ OpenAIRealtimeClient ready")
        except Exception as e:
            log(f"❌ OpenAIRealtimeClient init FAILED: {e}")
            log(traceback.format_exc())
            raise

        # ── Desktop audio streamer ────────────────────────────────────────────
        log(f"Initializing DesktopAudioStreamer (device_id={DESKTOP_AUDIO_DEVICE_ID}) …")
        try:
            self.streamer = DesktopAudioStreamer(
                realtime_client = self.openai_client,
                device_id       = DESKTOP_AUDIO_DEVICE_ID,
            )
            self.streamer.set_volume_callback(self._on_volume)
            log("✅ DesktopAudioStreamer ready")
        except Exception as e:
            log(f"❌ DesktopAudioStreamer init FAILED: {e}")
            log(traceback.format_exc())
            raise

        # ── Transcript enricher ───────────────────────────────────────────────
        self.enricher: TranscriptEnricher | None = None
        if ENABLE_ENRICHMENT:
            log("Initializing TranscriptEnricher …")
            try:
                self.enricher = TranscriptEnricher(
                    on_enriched_transcript = self._on_enriched_transcript,
                    on_audio_atmosphere    = self._on_audio_atmosphere,
                )
                log("✅ TranscriptEnricher ready")
            except Exception as e:
                log(f"❌ TranscriptEnricher init FAILED: {e}")
                log(traceback.format_exc())
                raise
        else:
            log("ℹ️  TranscriptEnricher disabled (ENABLE_ENRICHMENT=False)")

        log("✅ StreamAudioService initialized")

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self):
        self.hub_loop = asyncio.new_event_loop()
        threading.Thread(target=self._hub_thread, args=(self.hub_loop,), daemon=True, name="StreamHub").start()
        log("Hub thread started")

        self.ws_server.start()
        log("WebSocket server started")

        if self.enricher:
            self.enricher.start()
            log("Enricher started")

        self.streamer.start()
        log(f"DesktopAudioStreamer started — capturing device {DESKTOP_AUDIO_DEVICE_ID} …")

        log("✅ All components running — listening for desktop audio …")
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
        log(
            f"🛑 Shutting down ("
            f"hub emits: {self._hub_emit_count}, "
            f"WS: {self._ws_broadcast_count}, "
            f"whisper: {self._whisper_count}, "
            f"enriched: {self._enriched_count}, "
            f"atmosphere: {self._atmosphere_count})"
        )
        try:
            self.streamer.stop()
        except Exception as e:
            log(f"Error stopping streamer: {e}")
        if self.enricher:
            try:
                self.enricher.stop()
            except Exception as e:
                log(f"Error stopping enricher: {e}")
        try:
            self.ws_server.stop()
        except Exception as e:
            log(f"Error stopping WS server: {e}")
        if self.hub_loop:
            self.hub_loop.call_soon_threadsafe(self.hub_loop.stop)
        log("🛑 Stopped.")

    # ── Hub ───────────────────────────────────────────────────────────────────

    def _register_hub_events(self):
        @self.sio.event
        async def connect():
            log(f"✅ Hub CONNECTED → {HUB_URL}")

        @self.sio.event
        async def disconnect():
            log("⚠️  Hub DISCONNECTED")

        @self.sio.event
        async def connect_error(data):
            log(f"❌ Hub CONNECTION ERROR: {data}")

        @self.sio.on("vision_context")
        async def on_vision_context(data):
            self._vision_ctx_count += 1
            ctx = data.get("context", "")
            log(f"📥 HUB vision_context #{self._vision_ctx_count}: {repr(ctx[:80])}")
            if ctx and self.enricher:
                self.enricher.update_visual_context(ctx)
                log("  ↳ visual context forwarded to enricher")

        @self.sio.on("text_update")
        async def on_text_update(data):
            ctx = data.get("content", "")
            log(f"📥 HUB text_update: {repr(ctx[:80])}")
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
                    log(f"Attempting hub connect → {HUB_URL} …")
                    await self.sio.connect(HUB_URL)
                except Exception as e:
                    log(f"⚠️  Hub connect failed: {e} — retry in 5s")
                    await asyncio.sleep(5)
            await asyncio.sleep(2)

    def _emit_to_hub(self, event: str, data: dict):
        if not self.sio.connected:
            log(f"⚠️  SKIPPED hub emit (not connected): {event}")
            return
        if not self.hub_loop:
            log(f"⚠️  SKIPPED hub emit (no loop): {event}")
            return
        try:
            asyncio.run_coroutine_threadsafe(self.sio.emit(event, data), self.hub_loop)
            self._hub_emit_count += 1
            log(f"→ HUB [{event}] {str(data)[:160]}")
        except Exception as e:
            log(f"❌ HUB EMIT ERROR [{event}]: {e}")
            log(traceback.format_exc())

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_volume(self, level: float):
        self.ws_server.broadcast({"type": "volume", "source": "desktop", "level": level})

    def _on_whisper_transcript(self, raw_text: str):
        self._whisper_count += 1
        tid = str(uuid.uuid4())
        log(f"🎙️  WHISPER TRANSCRIPT #{self._whisper_count}: {repr(raw_text)}")

        # Broadcast raw immediately
        try:
            self.ws_server.broadcast({
                "type":      "transcript_raw",
                "source":    "desktop",
                "text":      raw_text,
                "id":        tid,
                "timestamp": time.time(),
            })
            self._ws_broadcast_count += 1
            log(f"→ WS broadcast (raw) #{self._ws_broadcast_count}")
        except Exception as e:
            log(f"❌ WS BROADCAST ERROR (raw): {e}")
            log(traceback.format_exc())

        if self.enricher:
            log(f"  ↳ Sending to enricher (id={tid[:8]}…)")
            self.enricher.enrich(raw_text, transcript_id=tid)
        else:
            self._publish_transcript(raw_text, tid, enriched=False)

    def _on_enriched_transcript(
        self,
        enriched_text: str,
        transcript_id: str | None = None,
        atmosphere: dict | None = None,
    ):
        self._enriched_count += 1
        log(f"✨ ENRICHED TRANSCRIPT #{self._enriched_count}: {repr(enriched_text[:120])}")

        speaker = "Unknown"
        match = re.search(r"\[\d+:\d+\]\s*(?:\[.*?\]\s*)?([^:(]+?)(?:\s*\([^)]+\))?:", enriched_text)
        if match:
            speaker = match.group(1).strip()
        log(f"  ↳ Speaker detected: {repr(speaker)}")

        payload = {
            "type":       "transcript_enriched",
            "source":     "desktop",
            "speaker":    speaker,
            "text":       enriched_text,
            "enriched":   True,
            "id":         transcript_id,
            "timestamp":  time.time(),
        }
        if atmosphere:
            payload["atmosphere"] = atmosphere

        try:
            self.ws_server.broadcast(payload)
            self._ws_broadcast_count += 1
            log(f"→ WS broadcast (enriched) #{self._ws_broadcast_count}")
        except Exception as e:
            log(f"❌ WS BROADCAST ERROR (enriched): {e}")
            log(traceback.format_exc())

        hub_metadata = {"source": "desktop", "speaker": speaker, "id": transcript_id}
        if atmosphere:
            hub_metadata["atmosphere"] = atmosphere

        self._emit_to_hub("audio_context", {
            "context":    enriched_text,
            "is_partial": False,
            "timestamp":  time.time(),
            "metadata":   hub_metadata,
        })

        self._emit_to_hub("transcript_enriched", {
            "text":       enriched_text,
            "speaker":    speaker,
            "id":         transcript_id,
            "atmosphere": atmosphere or {},
        })

    def _on_audio_atmosphere(self, atmosphere: dict, transcript_id: str | None = None):
        """
        Fired whenever the enricher detects meaningful audio atmosphere.
        Broadcasts as a standalone event so consumers can react to music/SFX
        changes independently of transcript text.
        """
        self._atmosphere_count += 1
        mood    = atmosphere.get("mood", "")
        music   = atmosphere.get("music", "")
        changed = atmosphere.get("music_changed", False)

        log(
            f"🎵 ATMOSPHERE #{self._atmosphere_count} "
            f"[changed={changed}] mood={repr(mood)} music={repr(music[:60])}"
        )

        try:
            self.ws_server.broadcast({
                "type":       "audio_atmosphere",
                "source":     "desktop",
                "atmosphere": atmosphere,
                "id":         transcript_id,
                "timestamp":  time.time(),
            })
            self._ws_broadcast_count += 1
        except Exception as e:
            log(f"❌ WS BROADCAST ERROR (atmosphere): {e}")

        # Only emit to hub when something meaningfully changed
        if changed or self._atmosphere_count == 1:
            self._emit_to_hub("audio_atmosphere", {
                "atmosphere": atmosphere,
                "id":         transcript_id,
                "timestamp":  time.time(),
            })

    def _publish_transcript(self, text: str, tid: str, enriched: bool):
        log(f"→ Publishing transcript (enriched={enriched}): {repr(text[:80])}")
        try:
            self.ws_server.broadcast({
                "type":      "transcript",
                "source":    "desktop",
                "text":      text,
                "enriched":  enriched,
                "id":        tid,
                "timestamp": time.time(),
            })
            self._ws_broadcast_count += 1
        except Exception as e:
            log(f"❌ WS BROADCAST ERROR (publish): {e}")
            log(traceback.format_exc())

        self._emit_to_hub("audio_context", {
            "context":    text,
            "is_partial": False,
            "timestamp":  time.time(),
            "metadata":   {"source": "desktop", "id": tid},
        })

    def _on_openai_error(self, msg: str):
        log(f"❌ OPENAI ERROR: {msg}")
        self.ws_server.broadcast({"type": "error", "source": "openai", "message": msg})

    # ── Device hot-swap ───────────────────────────────────────────────────────

    def swap_device(self, device_id: int):
        """Stop the current streamer and restart it on a new device."""
        import config
        log(f"🔄 Device swap requested → device_id={device_id}")

        try:
            self.streamer.stop()
            log("  ↳ Old streamer stopped")
        except Exception as e:
            log(f"  ↳ Error stopping streamer: {e}")

        config.DESKTOP_AUDIO_DEVICE_ID = device_id

        try:
            from audio_streamer import DesktopAudioStreamer
            self.streamer = DesktopAudioStreamer(
                realtime_client = self.openai_client,
                device_id       = device_id,
            )
            self.streamer.set_volume_callback(self._on_volume)
            self.streamer.start()
            log(f"✅ New streamer started on device {device_id}")
        except Exception as e:
            log(f"❌ Failed to start streamer on device {device_id}: {e}")
            log(traceback.format_exc())