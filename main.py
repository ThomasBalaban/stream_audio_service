#!/usr/bin/env python3
"""
Stream Audio Service — Entry Point
====================================
Captures desktop audio → OpenAI Realtime (Whisper VAD) → GPT-4o enrichment.
Subscribes to Hub vision_context so the enricher always has visual context.

WebSocket clients:  ws://localhost:8017
Health check:       GET  http://localhost:8018/health
Shutdown:           POST http://localhost:8018/shutdown
"""

import os
import signal
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_THIS_DIR)

import http_control
from service import StreamAudioService

_service: StreamAudioService | None = None


def _shutdown(*_):
    global _service
    if _service:
        _service.stop()
    sys.exit(0)


def main():
    global _service
    _service = StreamAudioService()

    http_control.start(shutdown_callback=_shutdown)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    _service.run()


if __name__ == "__main__":
    main()