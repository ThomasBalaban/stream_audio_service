"""
Configuration for the Stream Audio Service.
Handles desktop audio capture → OpenAI Realtime transcription → GPT-4o enrichment.
"""
import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR  = os.path.dirname(_THIS_DIR)
_DESKTOP_DIR = os.path.join(_PARENT_DIR, "desktop_mon_gemini")

for _p in (_DESKTOP_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from api_keys import OPENAI_API_KEY  # noqa: E402

# Audio
DESKTOP_AUDIO_DEVICE_ID = 4
AUDIO_SAMPLE_RATE       = 16000

# OpenAI Realtime VAD settings
VAD_THRESHOLD            = 0.35
VAD_PREFIX_PADDING_MS    = 300
VAD_SILENCE_DURATION_MS  = 600
FORCE_COMMIT_INTERVAL_S  = 3.0

# Enrichment
ENABLE_ENRICHMENT = True   # Set False to skip GPT-4o enrichment step

# Network
WEBSOCKET_PORT    = 8017
HTTP_CONTROL_PORT = 8018
HUB_URL           = "http://localhost:8002"

# Vision service WebSocket — enricher subscribes to get visual context
VISION_WS_URL = "ws://localhost:8015"

SERVICE_NAME = "stream_audio_service"