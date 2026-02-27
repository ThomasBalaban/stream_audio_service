"""
HTTP control server for Stream Audio Service (port 8018).

GET  /health         → {"status": "ok"}
GET  /devices        → {"devices": [...], "current_device_id": N}
POST /set-device     → body: {"device_id": N}  → hot-swaps the desktop audio device
POST /shutdown       → clean shutdown
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import sounddevice as sd

from config import HTTP_CONTROL_PORT, SERVICE_NAME

_shutdown_cb    = None
_set_device_cb  = None
_server: HTTPServer | None = None


def _list_input_devices():
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({
                "id":       i,
                "name":     dev["name"],
                "channels": dev["max_input_channels"],
                "default_samplerate": int(dev["default_samplerate"]),
            })
    return devices


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "service": SERVICE_NAME, "port": HTTP_CONTROL_PORT})

        elif self.path == "/devices":
            try:
                import config as _config
                devices = _list_input_devices()
                self._json(200, {
                    "devices":           devices,
                    "current_device_id": _config.DESKTOP_AUDIO_DEVICE_ID,
                })
            except Exception as e:
                self._json(500, {"error": str(e)})

        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length) if length else b"{}"

        if self.path == "/shutdown":
            self._json(200, {"status": "shutting_down"})
            if _shutdown_cb:
                threading.Thread(target=_shutdown_cb, daemon=True).start()

        elif self.path == "/set-device":
            try:
                payload   = json.loads(body)
                device_id = int(payload["device_id"])
                if _set_device_cb:
                    threading.Thread(target=_set_device_cb, args=(device_id,), daemon=True).start()
                self._json(200, {"status": "ok", "device_id": device_id})
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": f"bad request: {e}"})
            except Exception as e:
                self._json(500, {"error": str(e)})

        else:
            self._json(404, {"error": "not found"})


def start(shutdown_callback, set_device_callback=None):
    global _shutdown_cb, _set_device_cb, _server
    _shutdown_cb   = shutdown_callback
    _set_device_cb = set_device_callback
    _server = HTTPServer(("0.0.0.0", HTTP_CONTROL_PORT), _Handler)
    t = threading.Thread(target=_server.serve_forever, daemon=True, name="StreamHTTP")
    t.start()
    print(f"✅ [StreamHTTP] Control server on :{HTTP_CONTROL_PORT} (/health /devices /set-device /shutdown)", flush=True)


def stop():
    global _server
    if _server:
        _server.shutdown()
        _server = None