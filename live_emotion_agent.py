"""Process 2: Gemini Live audio/context controller for WebPulse.

The active runtime uses one persistent Gemini Live WebSocket. The microphone
and speaker are owned by this process; camera/WESAD state arrives through the
localhost state broker. Legacy Whisper, HTTP LLM, and pyttsx3 paths are not
used here because they add a second turn-generation pipeline and latency.
"""

import asyncio
import base64
import collections
import json
import os
import threading
import time

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

from audio.valence import estimate_voice_valence, extract_audio_features
from fusion.emotion import fuse_emotions
from logging_.session_logger import SessionLogger


load_dotenv()

INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
CHUNK_FRAMES = 640  # 40 ms of microphone audio
MIC_QUEUE_CHUNKS = 8  # at most 320 ms of queued input
STATE_INTERVAL_SECONDS = 1.0
PHYSIOLOGY_MEMORY_SIZE = 10
MODEL_NAME = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
VOICE_NAME = os.getenv("GEMINI_LIVE_VOICE", "Puck")
WS_ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

SYSTEM_INSTRUCTION = """
You are WebPulse Live, an empathetic spoken companion for HCI research at
Midori Sugaya Sensei's Lab, Shibaura Institute of Technology. Speak briefly,
naturally, and directly about what the user says. Treat camera and voice
telemetry as uncertain context, never as medical facts or a diagnosis.

Background messages are labeled PHYSIOLOGY CONTEXT and contain only bounded
0-5 scores and signal-quality labels. Do not answer background messages by
themselves. When the user speaks, use the latest context silently. If the
bio-state score is high or stress is STRESSED, use a slower rhythm, soothing
word choice, and a lower perceived pitch. If signal quality is WEAK_SIGNAL,
POOR_LIGHTING, or UNAVAILABLE, do not claim to know the user's physical or
emotional state and invite correction. Never say that you know how the user
feels from the sensors.

When answering a question about how the user feels, use the latest
ANSWER-TIME PHYSIOLOGY context as the primary physiological evidence. Treat
the score labels as the current state, not as a request to discuss telemetry.
""".strip()


class PCMPlayback:
    """Low-latency native-audio playback queue with immediate barge-in clear."""

    def __init__(self):
        self._chunks = collections.deque()
        self._offset = 0
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=OUTPUT_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_FRAMES,
            latency="low",
            callback=self._callback,
        )

    def start(self):
        self._stream.start()

    def enqueue(self, pcm_bytes):
        samples = np.frombuffer(pcm_bytes, dtype="<i2")
        if samples.size:
            with self._lock:
                self._chunks.append(samples.copy())

    def clear(self):
        with self._lock:
            self._chunks.clear()
            self._offset = 0

    def stop(self):
        self.clear()
        self._stream.stop()
        self._stream.close()

    def _callback(self, outdata, frames, time_info, status):
        del time_info, status
        outdata.fill(0)
        written = 0
        with self._lock:
            while written < frames and self._chunks:
                chunk = self._chunks[0]
                count = min(frames - written, len(chunk) - self._offset)
                outdata[written:written + count, 0] = chunk[self._offset:self._offset + count]
                written += count
                self._offset += count
                if self._offset >= len(chunk):
                    self._chunks.popleft()
                    self._offset = 0


class MicrophoneStream:
    """Capture 16 kHz PCM and expose a short rolling analysis buffer."""

    def __init__(self, loop):
        self._loop = loop
        self.audio_queue = asyncio.Queue(maxsize=MIC_QUEUE_CHUNKS)
        self.last_speech_at = 0.0
        self._samples = collections.deque(maxlen=INPUT_RATE * 4)
        self._samples_lock = threading.Lock()
        self._stream = sd.InputStream(
            samplerate=INPUT_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_FRAMES,
            latency="low",
            callback=self._callback,
        )

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def recent_float_audio(self):
        with self._samples_lock:
            return np.asarray(self._samples, dtype=np.float32) / 32768.0

    def _callback(self, indata, frames, time_info, status):
        del frames, time_info, status
        pcm = indata.copy().astype("<i2", copy=False)
        rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        if rms > 450.0:
            self.last_speech_at = time.monotonic()
        with self._samples_lock:
            self._samples.extend(pcm[:, 0].tolist())
        self._loop.call_soon_threadsafe(self._put_chunk, pcm.tobytes())

    def _put_chunk(self, pcm):
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.audio_queue.put_nowait(pcm)


class BrokerStateSubscriber:
    """Receive latest-only physiology state from Process 1 via Process 3."""

    def __init__(self, queue):
        self.queue = queue

    async def run(self):
        while True:
            writer = None
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", 5003)
                writer.write(b'{"type":"subscribe"}\n')
                await writer.drain()
                print("[Live Agent] Connected to state broker (127.0.0.1:5003).")
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        state = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if self.queue.full():
                        try:
                            self.queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    self.queue.put_nowait(state)
            except (ConnectionError, OSError):
                await asyncio.sleep(1.0)
            finally:
                if writer is not None:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except OSError:
                        pass


class HUDClient:
    """Send transcripts and companion state to the camera HUD."""

    def __init__(self):
        self.writer = None
        self.lock = asyncio.Lock()

    async def run(self):
        while True:
            try:
                reader, self.writer = await asyncio.open_connection("127.0.0.1", 5001)
                print("[Live Agent] Connected to HUD server (127.0.0.1:5001).")
                while await reader.readline():
                    pass
            except (ConnectionError, OSError):
                await asyncio.sleep(1.0)
            finally:
                self.writer = None

    async def update(self, payload):
        if self.writer is None:
            return
        async with self.lock:
            try:
                self.writer.write((json.dumps(dict(payload, type="hud_update")) + "\n").encode("utf-8"))
                await self.writer.drain()
            except (ConnectionError, OSError):
                self.writer = None


class GeminiLiveEmotionAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is missing from .env")
        self.state_queue = asyncio.Queue(maxsize=1)
        self.current_state = {
            "heart_rate": None,
            "hrv": None,
            "arousal": 0.5,
            "stress_state": "NORMAL",
            "quality_status": "WEAK_SIGNAL",
            "valence": 0.0,
            "roi_count": 0,
        }
        self.microphone = None
        self.playback = None
        self.broker = BrokerStateSubscriber(self.state_queue)
        self.hud = HUDClient()
        self.session_logger = SessionLogger()
        self._send_lock = asyncio.Lock()
        self._model_speaking = False
        self._speech_active = False
        self._last_state_sent = 0.0
        self._last_state_signature = None
        self._physiology_memory = collections.deque(maxlen=PHYSIOLOGY_MEMORY_SIZE)
        self._last_memory_append = 0.0
        self._input_text = ""
        self._output_text = ""

    @property
    def ws_url(self):
        return f"{WS_ENDPOINT}?key={self.api_key}"

    def setup_payload(self):
        return {
            "setup": {
                "model": f"models/{MODEL_NAME}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE_NAME}}},
                },
                "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "realtimeInputConfig": {"automaticActivityDetection": {
                    "disabled": False,
                    "prefixPaddingMs": 20,
                    "silenceDurationMs": 300,
                }},
            }
        }

    async def run(self):
        if sd is None:
            raise RuntimeError("sounddevice is required for Gemini Live microphone/speaker I/O")
        loop = asyncio.get_running_loop()
        self.microphone = MicrophoneStream(loop)
        self.playback = PCMPlayback()
        self.session_logger.start_session(subject_id=os.getenv("WEBPULSE_SUBJECT", "subject_pilot"))
        self.microphone.start()
        self.playback.start()
        broker_task = asyncio.create_task(self.broker.run())
        hud_task = asyncio.create_task(self.hud.run())
        try:
            while True:
                try:
                    async with websockets.connect(self.ws_url, max_size=8 * 1024 * 1024) as ws:
                        await ws.send(json.dumps(self.setup_payload()))
                        await self._wait_for_setup(ws)
                        print(f"[Live Agent] Gemini Live connected ({MODEL_NAME}, {VOICE_NAME}).")
                        await self._run_session(ws)
                except Exception as exc:
                    print(f"[Live Agent] WebSocket disconnected: {exc}. Retrying in 2s.")
                    await asyncio.sleep(2.0)
        finally:
            broker_task.cancel()
            hud_task.cancel()
            await asyncio.gather(broker_task, hud_task, return_exceptions=True)
            self.microphone.stop()
            self.playback.stop()
            summary = self.session_logger.stop_session()
            if summary:
                print(f"[Session Logger] Exported session -> {summary['json_path']} & {summary['csv_path']}")

    async def _wait_for_setup(self, ws):
        while True:
            message = json.loads(await ws.recv())
            if "setupComplete" in message:
                return
            if "error" in message:
                raise RuntimeError(message["error"])

    async def _run_session(self, ws):
        tasks = [
            asyncio.create_task(self._send_microphone(ws)),
            asyncio.create_task(self._send_background_state(ws)),
            asyncio.create_task(self._receive_server(ws)),
            asyncio.create_task(self._update_valence()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in done:
            if task.exception():
                raise task.exception()

    async def _send(self, ws, payload):
        async with self._send_lock:
            await ws.send(json.dumps(payload))

    async def _send_microphone(self, ws):
        while True:
            pcm = await self.microphone.audio_queue.get()
            now = time.monotonic()
            speaking = now - self.microphone.last_speech_at < 0.18

            # Do not feed speaker playback back into Gemini. A real local
            # speech event is still allowed through so Gemini can interrupt
            # the current response (barge-in).
            if self._model_speaking and not speaking:
                continue

            if speaking and not self._speech_active:
                self._speech_active = True
                self._input_text = ""
                await self.hud.update({"transcript": ""})
                await self._send_latest_context(ws, force=True, reason="speech_start")

            await self._send(ws, {"realtimeInput": {"audio": {
                "mimeType": "audio/pcm;rate=16000",
                "data": base64.b64encode(pcm).decode("ascii"),
            }}})

            if self._speech_active and not speaking and now - self.microphone.last_speech_at >= 0.18:
                self._speech_active = False
                await self._send_latest_context(ws, force=True, reason="speech_end")

    async def _send_background_state(self, ws):
        while True:
            self.current_state.update(await self.state_queue.get())
            self._remember_current_state()
            now = time.monotonic()
            if self._model_speaking or now - self.microphone.last_speech_at < 1.0:
                continue
            # Keep physiology in the local memory buffer while idle. Sending
            # telemetry as realtimeInput text can itself be interpreted as a
            # conversational turn and cause unsolicited model speech.

    async def _send_latest_context(self, ws, force=False, reason="periodic"):
        now = time.monotonic()
        if not force and now - self._last_state_sent < STATE_INTERVAL_SECONDS:
            return
        context = self._latest_memory_context()
        signature = tuple(sorted(context.items()))
        if not force and signature == self._last_state_signature:
            return
        self._last_state_signature = signature
        self._last_state_sent = now
        message_label = "ANSWER-TIME PHYSIOLOGY" if reason in {"speech_start", "speech_end"} else "PHYSIOLOGY CONTEXT"
        text = f"{message_label} ({reason}; do not answer): {json.dumps(context, separators=(',', ':'))}"
        # Metadata is deliberately sent as an incomplete client turn. It
        # updates context without creating a fresh user turn or allowing the
        # model to answer telemetry by itself. User audio remains the only
        # trigger for a response.
        await self._send(ws, {"clientContent": {"turns": [{"role": "user", "parts": [{
            "text": text
        }]}], "turnComplete": False}})
        fused = self._fused_from_context(context)
        await self.hud.update({
            "valence": fused["valence"],
            "emotion_label": fused["label"],
            "emotion_desc": fused["description"],
            "stress_label": fused["stress_label"],
            "roi_count": self.current_state.get("roi_count", 0),
        })

    def _remember_current_state(self):
        """Append one scored snapshot per second to a fixed-size memory."""
        now = time.monotonic()
        if now - self._last_memory_append < STATE_INTERVAL_SECONDS:
            return
        fused = fuse_emotions(
            self.current_state.get("arousal"), self.current_state.get("valence"),
            hrv_rmssd=self.current_state.get("hrv"),
            heart_rate=self.current_state.get("heart_rate") or 70.0,
            stress_label=self.current_state.get("stress_state"),
            classifier_source="process_1_wesad",
        )
        context = self._build_scored_context(fused)
        self._physiology_memory.append({
            "timestamp": time.time(),
            "scores": context,
        })
        self._last_memory_append = now

    def _latest_memory_context(self):
        if not self._physiology_memory:
            self._remember_current_state()
        if self._physiology_memory:
            return dict(self._physiology_memory[-1]["scores"])
        return self._build_scored_context(self._fused_from_current_state())

    def _fused_from_current_state(self):
        return fuse_emotions(
            self.current_state.get("arousal"), self.current_state.get("valence"),
            hrv_rmssd=self.current_state.get("hrv"),
            heart_rate=self.current_state.get("heart_rate") or 70.0,
            stress_label=self.current_state.get("stress_state"),
            classifier_source="process_1_wesad",
        )

    def _fused_from_context(self, context):
        """Provide HUD labels while keeping the model payload score-only."""
        fused = self._fused_from_current_state()
        if context.get("signal_quality") == "UNAVAILABLE":
            fused["label"] = "UNAVAILABLE"
        return fused

    def _build_scored_context(self, fused):
        def score(value, low, high, invert=False, default=3.0):
            if value is None:
                return default
            normalized = float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))
            if invert:
                normalized = 1.0 - normalized
            return round(1.0 + 4.0 * normalized, 1)

        arousal = float(np.clip(fused.get("arousal", 0.5), 0.0, 1.0))
        valence = float(np.clip(fused.get("valence", 0.0), -1.0, 1.0))
        stress = str(fused.get("stress_label", "NORMAL")).upper()
        quality = str(self.current_state.get("quality_status", "WEAK_SIGNAL")).upper()
        stress_score = {"CALM": 1.0, "NORMAL": 3.0, "STRESSED": 5.0}.get(stress, 3.0)
        arousal_score = round(1.0 + 4.0 * arousal, 1)
        bio_score = round(0.7 * stress_score + 0.3 * arousal_score, 1) if quality == "GOOD" else 0.0
        return {
            "bio_state_score_5": bio_score,
            "bio_state_label": "UNAVAILABLE" if bio_score == 0.0 else "ACTIVE",
            "wesad_stress_label": stress,
            "wesad_stress_score_5": stress_score,
            "arousal_score_5": arousal_score,
            "valence_score_5": round(1.0 + 2.0 * (valence + 1.0), 1),
            "hr_score_5": score(self.current_state.get("heart_rate"), 50.0, 120.0),
            "hrv_stress_load_5": score(self.current_state.get("hrv"), 10.0, 100.0, invert=True),
            "signal_quality": quality,
            "roi_coverage": f"{self.current_state.get('roi_count', 0)}/3",
        }

    async def _update_valence(self):
        while True:
            await asyncio.sleep(2.0)
            samples = self.microphone.recent_float_audio()
            if len(samples) >= INPUT_RATE:
                features = await asyncio.to_thread(extract_audio_features, samples, INPUT_RATE)
                self.current_state["valence"] = float(estimate_voice_valence(features)) if features.get("has_speech") else 0.0

    async def _receive_server(self, ws):
        async for raw in ws:
            message = json.loads(raw)
            content = message.get("serverContent", {})
            if content.get("interrupted"):
                self._model_speaking = False
                self.playback.clear()
                await self.hud.update({"llm_response": ""})
                continue

            input_text = self._read_transcript(content, "inputTranscription")
            if input_text:
                self._input_text = self._merge_text(self._input_text, input_text)
                await self.hud.update({"transcript": self._input_text})

            output_text = self._read_transcript(content, "outputTranscription")
            if output_text:
                self._output_text = self._merge_text(self._output_text, output_text)
                await self.hud.update({"llm_response": self._output_text})

            for part in content.get("modelTurn", {}).get("parts", []):
                inline = part.get("inlineData") or {}
                if inline.get("data"):
                    self._model_speaking = True
                    self.playback.enqueue(base64.b64decode(inline["data"]))

            if content.get("turnComplete"):
                self._model_speaking = False
                if self._input_text or self._output_text:
                    fused = fuse_emotions(
                        self.current_state.get("arousal"), self.current_state.get("valence"),
                        hrv_rmssd=self.current_state.get("hrv"),
                        heart_rate=self.current_state.get("heart_rate") or 70.0,
                        stress_label=self.current_state.get("stress_state"),
                        classifier_source="process_1_wesad",
                    )
                    self.session_logger.log_entry(
                        self.current_state.get("heart_rate"), self.current_state.get("hrv"),
                        fused["arousal"], fused["valence"], fused["label"],
                        self._input_text, self._output_text,
                        raw_signal_summary=self._build_scored_context(fused),
                    )
                self._output_text = ""

    @staticmethod
    def _read_transcript(content, field):
        value = content.get(field)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            return value["text"].strip()
        return ""

    @staticmethod
    def _merge_text(current, addition):
        if not current:
            return addition
        if addition.startswith(current):
            return addition
        if current.endswith(addition):
            return current
        return f"{current} {addition}".strip()


def main():
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(GeminiLiveEmotionAgent().run())
    except KeyboardInterrupt:
        print("\n[Live Agent] Stopped.")


if __name__ == "__main__":
    main()
