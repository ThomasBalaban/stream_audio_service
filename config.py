"""
Configuration for the Stream Audio Service.
Handles desktop audio capture → OpenAI Realtime transcription → GPT-4o enrichment.
"""
from api_keys import OPENAI_API_KEY

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