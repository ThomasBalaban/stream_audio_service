"""
Configuration for the Stream Audio Service.
Handles desktop audio capture → OpenAI Realtime transcription → GPT-4o enrichment.
"""
import os
import sys
from pathlib import Path

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

print(f"[config] sys.path[0:3] = {sys.path[:3]}", flush=True)


def _load_sibling_secrets() -> None:
    secrets_dir = Path(__file__).resolve().parent.parent / "director_ui" / "secrets"
    if not secrets_dir.is_dir():
        return
    for path in sorted(secrets_dir.glob("*.env")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if "OPENAI_API_KEY" not in os.environ:
    _load_sibling_secrets()

try:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
except KeyError as e:
    raise RuntimeError(
        f"Missing required env var {e.args[0]}. "
        "Credentials live in director_ui/secrets/*.env. Start this service via "
        "the launcher, or export the variable manually."
    ) from None

# Audio
AUDIO_SAMPLE_RATE      = 16000

_PREFERRED_DEVICE_NAME = "Cam Link 4K"
_FALLBACK_DEVICE_ID    = 4

def _find_device_by_name(name: str) -> int | None:
    try:
        import sounddevice as sd
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and name.lower() in dev["name"].lower():
                print(f"[config] ✅ Found '{name}' → device_id={i} ({dev['name']})", flush=True)
                return i
    except Exception as e:
        print(f"[config] ⚠️  Device lookup failed: {e}", flush=True)
    return None

_detected = _find_device_by_name(_PREFERRED_DEVICE_NAME)
if _detected is None:
    print(f"[config] ⚠️  '{_PREFERRED_DEVICE_NAME}' not found — falling back to device_id={_FALLBACK_DEVICE_ID}", flush=True)
DESKTOP_AUDIO_DEVICE_ID = _detected if _detected is not None else _FALLBACK_DEVICE_ID

# OpenAI Realtime VAD
VAD_THRESHOLD           = 0.35
VAD_PREFIX_PADDING_MS   = 300
VAD_SILENCE_DURATION_MS = 600
FORCE_COMMIT_INTERVAL_S = 3.0

# Enrichment
ENABLE_ENRICHMENT = True

# Network
WEBSOCKET_PORT    = 8017
HTTP_CONTROL_PORT = 8018
HUB_URL           = "http://localhost:8002"

VISION_WS_URL = "ws://localhost:8015"
SERVICE_NAME  = "stream_audio_service"