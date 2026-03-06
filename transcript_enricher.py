"""
TranscriptEnricher for Stream Audio Service.
Uses GPT-4o to add speaker labels, tone markers, timestamps,
and dynamic audio atmosphere detection (music mood, SFX, ambience).
Visual context is fed in from the vision_service via Hub subscription.
"""

import json
import re
import threading
import time

from openai import OpenAI

from config import OPENAI_API_KEY


class TranscriptEnricher:
    def __init__(self, on_enriched_transcript, on_audio_atmosphere=None):
        self.client              = OpenAI(api_key=OPENAI_API_KEY)
        self.on_enriched         = on_enriched_transcript
        self.on_audio_atmosphere = on_audio_atmosphere  # Optional separate callback

        # Context fed by vision_service
        self.visual_context      = ""

        # Recent transcripts for continuity
        self.recent_transcripts: list[str] = []
        self.max_history         = 8

        # Recent atmosphere for continuity (avoid repeating identical annotations)
        self.last_atmosphere: dict = {}

        # Known speakers → consistent labels
        self.known_speakers: dict[str, str] = {}
        self.speaker_counter = {"female": 0, "male": 0, "unknown": 0}

        self.session_start = time.time()

        # Processing queue
        self._queue: list[dict]        = []
        self._lock                     = threading.Lock()
        self.running                   = False
        self._thread: threading.Thread | None = None

    # ── Public ───────────────────────────────────────────────────────────────

    def start(self):
        self.running       = True
        self.session_start = time.time()
        self._thread       = threading.Thread(
            target=self._loop, daemon=True, name="Enricher"
        )
        self._thread.start()
        print("🎭 [Enricher] Started (GPT-4o speaker + audio atmosphere tracking)")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    def update_visual_context(self, context: str):
        """Called when the vision_service broadcasts a new analysis."""
        self.visual_context = context

    def enrich(self, raw_transcript: str, transcript_id: str | None = None):
        if not raw_transcript or len(raw_transcript.strip()) < 2:
            return
        with self._lock:
            self._queue.append({
                "text":           raw_transcript,
                "timestamp":      time.time() - self.session_start,
                "visual_context": self.visual_context,
                "id":             transcript_id,
            })

    # ── Internal ─────────────────────────────────────────────────────────────

    def _loop(self):
        while self.running:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.pop(0)
            if item:
                try:
                    result = self._enrich(item)
                    if result:
                        enriched_line = result["formatted_line"]
                        atmosphere    = result["atmosphere"]

                        if enriched_line and self.on_enriched:
                            self.on_enriched(enriched_line, item.get("id"), atmosphere)

                        if atmosphere and self.on_audio_atmosphere:
                            self.on_audio_atmosphere(atmosphere, item.get("id"))

                        self.last_atmosphere = atmosphere

                except Exception as e:
                    print(f"⚠️  [Enricher] Error: {e}")
                    if self.on_enriched:
                        ts = self._fmt_ts(item["timestamp"])
                        self.on_enriched(f"[{ts}] {item['text']}", item.get("id"), {})
            else:
                time.sleep(0.1)

    def _fmt_ts(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    def _speaker_history(self) -> str:
        if not self.known_speakers:
            return "No speakers identified yet."
        lines = ["Previously identified speakers:"]
        for desc, label in self.known_speakers.items():
            lines.append(f"  - {label}: {desc}")
        return "\n".join(lines)

    def _atmosphere_history(self) -> str:
        if not self.last_atmosphere:
            return "No prior audio atmosphere detected."
        parts = []
        if self.last_atmosphere.get("music"):
            parts.append(f"Music: {self.last_atmosphere['music']}")
        if self.last_atmosphere.get("sfx"):
            parts.append(f"SFX: {', '.join(self.last_atmosphere['sfx'])}")
        if self.last_atmosphere.get("ambience"):
            parts.append(f"Ambience: {self.last_atmosphere['ambience']}")
        return "Prior atmosphere: " + " | ".join(parts) if parts else "No prior audio atmosphere."

    def _enrich(self, item: dict) -> dict | None:
        raw    = item["text"]
        ts     = self._fmt_ts(item["timestamp"])
        visual = item["visual_context"] or "No visual context available"
        history = ""
        if self.recent_transcripts:
            history = "Recent transcript history (for continuity):\n"
            history += "\n".join(self.recent_transcripts[-5:]) + "\n"

        prompt = f"""You are a professional audio/transcript analyst for media content.

CURRENT VISUAL CONTEXT:
{visual}

{self._speaker_history()}

{self._atmosphere_history()}

{history}
RAW AUDIO TRANSCRIPT:
"{raw}"

TIMESTAMP: [{ts}]

TASK: Analyze this audio segment and return a JSON object with exactly these fields:

{{
  "formatted_line": "<[timestamp] Speaker Name (tone/delivery): dialogue text>",
  "atmosphere": {{
    "music": "<Describe the music if present — its emotional quality, genre feel, tempo, instrumentation — e.g. 'tense low strings building to a crescendo', 'lo-fi hip hop beats, warm and melancholic', 'silence'. Infer from context if not explicit. null if definitely no music.>",
    "sfx": ["<specific sound effect>", "..."],
    "ambience": "<overall sonic environment — e.g. 'crowded diner, clinking cutlery', 'dead silent room', 'outdoor wind'. null if unclear.>",
    "mood": "<single evocative word or short phrase that captures the audio atmosphere — e.g. 'ominous', 'lighthearted', 'romantic tension', 'chaotic'>",
    "music_changed": <true if music/atmosphere has clearly shifted from prior context, false otherwise>
  }}
}}

RULES:
- formatted_line: include speaker, tone in parentheses (whispering), (laughing), (urgent), SFX inline as [SFX: gunshot] if they interrupt speech
- music: describe what you HEAR or can strongly infer — never hardcode genre names blindly, describe the feel
- sfx: only list discrete, distinct sound events (not continuous ambience)
- If no speech, formatted_line can be just atmosphere e.g. "[0:42] [SFX: door slam] [Music: tension spike]"
- Return ONLY valid JSON, no markdown fences, no extra text"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role":    "system",
                    "content": (
                        "You are an expert audio analyst and transcript formatter. "
                        "Return only valid JSON as specified. No markdown, no explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw_json = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as e:
            print(f"⚠️  [Enricher] JSON parse failed: {e} — raw: {raw_json[:200]}")
            return {
                "formatted_line": f"[{ts}] {raw}",
                "atmosphere":     {},
            }

        formatted_line = parsed.get("formatted_line", f"[{ts}] {raw}")
        atmosphere     = parsed.get("atmosphere", {})

        # Normalise sfx to always be a list
        if isinstance(atmosphere.get("sfx"), str):
            atmosphere["sfx"] = [atmosphere["sfx"]] if atmosphere["sfx"] else []
        elif not isinstance(atmosphere.get("sfx"), list):
            atmosphere["sfx"] = []

        # Strip null values for cleaner downstream payloads
        atmosphere = {k: v for k, v in atmosphere.items() if v is not None and v != []}

        self._track_speaker(formatted_line)
        self.recent_transcripts.append(formatted_line)
        if len(self.recent_transcripts) > self.max_history:
            self.recent_transcripts.pop(0)

        # Log atmosphere if interesting
        if atmosphere.get("mood"):
            print(f"🎵 [Enricher] Atmosphere — mood: {atmosphere.get('mood')} | "
                  f"music: {atmosphere.get('music', '—')[:60]}")

        return {"formatted_line": formatted_line, "atmosphere": atmosphere}

    def _track_speaker(self, line: str):
        match = re.search(
            r"\[\d+:\d+\]\s*(?:\[.*?\]\s*)?([^:(]+?)(?:\s*\([^)]+\))?:", line
        )
        if match:
            speaker = match.group(1).strip()
            if any(x in speaker.lower() for x in
                   ["female", "male", "voice", "singer", "girl", "boy", "woman", "man"]):
                key = speaker.lower()
                if key not in self.known_speakers:
                    self.known_speakers[key] = speaker