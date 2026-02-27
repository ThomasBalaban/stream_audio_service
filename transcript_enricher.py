"""
TranscriptEnricher for Stream Audio Service.
Uses GPT-4o to add speaker labels, tone markers, and timestamps
to raw Whisper transcripts. Visual context is fed in from the
vision_service via Hub subscription in StreamAudioService.
"""

import re
import threading
import time

from openai import OpenAI

from config import OPENAI_API_KEY


class TranscriptEnricher:
    def __init__(self, on_enriched_transcript):
        self.client               = OpenAI(api_key=OPENAI_API_KEY)
        self.on_enriched          = on_enriched_transcript

        # Context fed by vision_service
        self.visual_context       = ""

        # Recent transcripts for continuity
        self.recent_transcripts: list[str] = []
        self.max_history          = 8

        # Known speakers → consistent labels
        self.known_speakers: dict[str, str] = {}
        self.speaker_counter      = {"female": 0, "male": 0, "unknown": 0}

        self.session_start        = time.time()

        # Processing queue
        self._queue: list[dict]   = []
        self._lock                = threading.Lock()
        self.running              = False
        self._thread: threading.Thread | None = None

    # ── Public ───────────────────────────────────────────────────────────────

    def start(self):
        self.running       = True
        self.session_start = time.time()
        self._thread       = threading.Thread(
            target=self._loop, daemon=True, name="Enricher"
        )
        self._thread.start()
        print("🎭 [Enricher] Started (GPT-4o speaker tracking)")

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
                    enriched = self._enrich(item)
                    if enriched and self.on_enriched:
                        self.on_enriched(enriched, item.get("id"))
                except Exception as e:
                    print(f"⚠️  [Enricher] Error: {e}")
                    if self.on_enriched:
                        ts = self._fmt_ts(item["timestamp"])
                        self.on_enriched(f"[{ts}] {item['text']}", item.get("id"))
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

    def _enrich(self, item: dict) -> str:
        raw       = item["text"]
        ts        = self._fmt_ts(item["timestamp"])
        visual    = item["visual_context"] or "No visual context available"
        history   = ""
        if self.recent_transcripts:
            history = "Recent transcript history (for continuity):\n"
            history += "\n".join(self.recent_transcripts[-5:]) + "\n"

        prompt = (
            f"You are a professional transcript formatter.\n\n"
            f"CURRENT VISUAL CONTEXT:\n{visual}\n\n"
            f"{self._speaker_history()}\n\n"
            f"{history}\n"
            f"RAW AUDIO:\n\"{raw}\"\n\n"
            f"TIMESTAMP: [{ts}]\n\n"
            "TASK: Format the raw transcription.\n"
            "- Identify SPEAKER (character name if known, else 'Male Voice 1' etc.)\n"
            "- Add TONE in parentheses: (sarcastic), (whispering)\n"
            "- Add SFX/Music if implied.\n"
            "OUTPUT ONLY the formatted line."
        )

        response  = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",  "content": "You are a transcript formatter. Output only the formatted line."},
                {"role": "user",    "content": prompt},
            ],
            max_tokens=250,
            temperature=0.3,
        )
        enriched = response.choices[0].message.content.strip()
        self._track_speaker(enriched)

        self.recent_transcripts.append(enriched)
        if len(self.recent_transcripts) > self.max_history:
            self.recent_transcripts.pop(0)

        return enriched

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