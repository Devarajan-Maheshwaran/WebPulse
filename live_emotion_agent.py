"""Process 2: Gemini Live audio and context controller for WebPulse."""

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


load_dotenv()

INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
CHUNK_FRAMES = 640  # 40 ms at 16 kHz
STATE_INTERVAL_SECONDS = 2.0
MAX_AUDIO_QUEUE_CHUNKS = 8  # 320 ms maximum stale microphone audio
# Keep both native transcript channels enabled so the HUD can distinguish the
# user's words from the companion's response. Audio generation is unaffected.
ENABLE_TRANSCRIPTIONS = os.getenv("GEMINI_LIVE_TRANSCRIPTIONS", "1") == "1"
MODEL_NAME = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
VOICE_NAME = os.getenv("GEMINI_LIVE_VOICE", "Puck")
WS_ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

SYSTEM_INSTRUCTION = """
You are WebPulse Live, a calm, supportive spoken companion for HCI research at
Midori Sugaya Sensei's Lab, Shibaura Institute of Technology. Speak naturally,
briefly, and empathetically. Treat camera-derived physiology and voice valence
as uncertain contextual signals, never as medical facts or a diagnosis.

Physiology messages use labeled scores from 1 to 5. A higher stress or arousal
score means more activation; a higher valence score means more positive affect;
a higher HRV stress-load score means a less calm signal. Do not infer raw
medical measurements from these scores.

Each context message also has `bio_state_score_5`: a single, explainable
activation/stress summary derived from the WESAD stress level and physiological
arousal. Score 0 means the visual signal is unreliable, 1 is very calm, 3 is
moderate, and 5 is high activation/stress. Treat score 0 as unavailable rather
than calm and use cautious language whenever signal quality is not GOOD.

Physiology updates arrive continuously as background context. Do not reply to
those updates on their own; respond when the user speaks. When the latest state
shows high arousal, STRESSED status, or negative valence, use a soothing tone,
slower rhythm, simpler phrasing, and a lower perceived pitch. If signal quality
is WEAK_SIGNAL or POOR_LIGHTING, explicitly keep claims tentative. Prefer
changes from the user's own recent baseline over absolute physiological values.
Never tell a user that you know how they feel; invite correction and support
agency.
""".strip()


class PCMPlayback:
    """Low-latency 24 kHz speaker queue that is cleared on a barge-in."""

    def __init__(self):
        self._chunks = collections.deque()
        self._offset = 0
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=OUTPUT_RATE, channels=1, dtype="int16",
            blocksize=CHUNK_FRAMES, latency="low", callback=self._callback,
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
                if self._offset == len(chunk):
                    self._chunks.popleft()
                    self._offset = 0


class MicrophoneStream:
    """The only microphone owner in the three-process design."""

    def __init__(self, loop):
        self.audio_queue = asyncio.Queue(maxsize=MAX_AUDIO_QUEUE_CHUNKS)
        self._loop = loop
        self.last_speech_at = 0.0
        self._samples = collections.deque(maxlen=INPUT_RATE * 4)
        self._samples_lock = threading.Lock()
        self._stream = sd.InputStream(
            samplerate=INPUT_RATE, channels=1, dtype="int16",
            blocksize=CHUNK_FRAMES, latency="low", callback=self._callback,
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
        if float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) > 450.0:
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
    """Receives latest-only physiology state from Process 3 without disk I/O."""

    def __init__(self, queue):
        self.queue = queue

    async def run(self):
        while True:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", 5003)
                writer.write(b'{"type":"subscribe"}\n')
                await writer.drain()
                print("[Live Agent] Connected to state broker.")
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        state = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    await self.queue.put(state)
            except (ConnectionError, OSError):
                await asyncio.sleep(1)
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                except (UnboundLocalError, OSError):
                    pass


class HUDClient:
    """Keeps the existing camera HUD connection separate from physiology IPC."""

    def __init__(self):
        self.writer = None
        self.lock = asyncio.Lock()

    async def run(self):
        while True:
            try:
                reader, self.writer = await asyncio.open_connection("127.0.0.1", 5001)
                while await reader.readline():
                    pass  # Discard metrics: Process 3 is the source of physiological state.
            except (ConnectionError, OSError):
                await asyncio.sleep(1)
            finally:
                self.writer = None

    async def update(self, payload):
        if self.writer is None:
            return
        async with self.lock:
            self.writer.write((json.dumps(dict(payload, type="hud_update")) + "\n").encode("utf-8"))
            await self.writer.drain()


class GeminiLiveEmotionAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is missing from .env")
        self.state_queue = asyncio.Queue()
        self.current_state = {
            "heart_rate": None, "hrv": None, "arousal": 0.5,
            "stress_state": "NORMAL", "quality_status": "WEAK_SIGNAL", "valence": 0.0,
        }
        self.playback = None
        self.microphone = None
        self.hud = HUDClient()
        self.broker = BrokerStateSubscriber(self.state_queue)
        self._model_speaking = False
        self._last_state_sent = 0.0
        self._last_context_signature = None
        self._last_response_latency_log = 0.0
        self._send_lock = asyncio.Lock()
        self._speech_active = False

    @property
    def ws_url(self):
        return f"{WS_ENDPOINT}?key={self.api_key}"

    def setup_payload(self):
        payload = {
            "setup": {
                "model": f"models/{MODEL_NAME}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE_NAME}}},
                },
                "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
                "realtimeInputConfig": {"automaticActivityDetection": {
                    "disabled": False, "prefixPaddingMs": 20, "silenceDurationMs": 300,
                }},
            }
        }
        if ENABLE_TRANSCRIPTIONS:
            payload["setup"].update({
                "inputAudioTranscription": {}, "outputAudioTranscription": {},
            })
        return payload

    async def run(self):
        loop = asyncio.get_running_loop()
        self.playback = PCMPlayback()
        self.microphone = MicrophoneStream(loop)
        self.playback.start()
        self.microphone.start()
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
                    print(f"[Live Agent] Gemini session disconnected: {exc}. Retrying in 2s.")
                    await asyncio.sleep(2)
        finally:
            for task in (broker_task, hud_task):
                task.cancel()
            await asyncio.gather(broker_task, hud_task, return_exceptions=True)
            self.microphone.stop()
            self.playback.stop()

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

            # Push the latest physiology before Gemini's VAD can finish the
            # user's turn. This makes every speech turn use current body data,
            # instead of depending on the periodic background timer.
            now = time.monotonic()
            speaking = now - self.microphone.last_speech_at < 0.18
            if speaking and not self._speech_active:
                self._speech_active = True
                await self._send_latest_state(ws, force=True, reason="speech_start")

            await self._send(ws, {"realtimeInput": {"audio": {
                "mimeType": "audio/pcm;rate=16000",
                "data": base64.b64encode(pcm).decode("ascii"),
            }}})

            # Send once more after speech drops below the local activity
            # threshold. Gemini's server VAD still has its silence window, so
            # this update arrives before turnComplete and response generation.
            if self._speech_active and not speaking and now - self.microphone.last_speech_at >= 0.18:
                self._speech_active = False
                await self._send_latest_state(ws, force=True, reason="speech_end")

    async def _send_background_state(self, ws):
        while True:
            self.current_state.update(await self.state_queue.get())
            now = time.monotonic()
            if (
                self._model_speaking
                or now - self.microphone.last_speech_at < 1.0
                or now - self._last_state_sent < STATE_INTERVAL_SECONDS
            ):
                continue
            await self._send_latest_state(ws)

    async def _send_latest_state(self, ws, force=False, reason="periodic"):
        """Inject current physiology without completing the user's turn."""
        now = time.monotonic()
        if not force and now - self._last_state_sent < STATE_INTERVAL_SECONDS:
            return

        fused = fuse_emotions(
            self.current_state.get("arousal"), self.current_state.get("valence"),
            hrv_rmssd=self.current_state.get("hrv"), heart_rate=self.current_state.get("heart_rate") or 70.0,
            stress_label=self.current_state.get("stress_state"),
            classifier_source="process_1_wesad_temporal",
        )
        physiological_context = self._build_scored_context(fused)
        context = {
            **physiological_context,
            "signal_quality": self.current_state.get("quality_status"),
        }
        signature = (
            context["stress"], context["signal_quality"],
            context["stress_score_5"], context["arousal"]["score_5"],
            context["valence"]["score_5"], context["heart_rate"]["score_5"],
            context["hrv_stress_load"]["score_5"], context["bio_state_score_5"],
        )
        # Periodic updates can be deduplicated. Speech-turn updates cannot:
        # the same physiology still needs to be attached to each new utterance.
        if not force and signature == self._last_context_signature:
            return

        self._last_state_sent = now
        self._last_context_signature = signature
        await self._send(ws, {"clientContent": {"turns": [{"role": "user", "parts": [{
            "text": f"BACKGROUND PHYSIOLOGY UPDATE ({reason}; do not answer yet): " + json.dumps(context)
        }]}], "turnComplete": False}})
        await self.hud.update({
            "valence": fused["valence"], "emotion_label": fused["label"],
            "emotion_desc": fused["description"], "stress_label": fused["stress_label"],
        })

    def _build_scored_context(self, fused):
        """Return only labeled, bounded 0-5 context for Gemini.

        Raw BPM/RMSSD values remain local to the application and are not
        included in the model-facing background message.

        This lightweight ordinal rating is analogous to a physiological-state
        annotation scale: it improves interpretability without adding a model
        or changing the local rPPG/WESAD inference path.
        """
        def score_1_to_5(value, low, high, invert=False, default=3.0):
            if value is None:
                return default
            normalized = float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))
            if invert:
                normalized = 1.0 - normalized
            return round(1.0 + 4.0 * normalized, 1)

        arousal = float(np.clip(fused.get("arousal", 0.5), 0.0, 1.0))
        valence = float(np.clip(fused.get("valence", 0.0), -1.0, 1.0))
        stress = str(fused.get("stress_label", "NORMAL")).upper()
        heart_rate = self.current_state.get("heart_rate")
        rmssd = self.current_state.get("hrv")
        signal_quality = str(self.current_state.get("quality_status", "WEAK_SIGNAL")).upper()

        arousal_label = "LOW" if arousal < 0.30 else "MODERATE" if arousal < 0.70 else "HIGH"
        valence_label = "NEGATIVE" if valence <= -0.40 else "NEUTRAL" if valence < 0.40 else "POSITIVE"

        if heart_rate is None:
            heart_rate_label = "UNAVAILABLE"
        elif heart_rate < 60:
            heart_rate_label = "LOW"
        elif heart_rate > 100:
            heart_rate_label = "ELEVATED"
        else:
            heart_rate_label = "TYPICAL"

        if rmssd is None:
            hrv_label = "UNAVAILABLE"
        elif rmssd < 25:
            hrv_label = "LOW_HRV"
        elif rmssd > 50:
            hrv_label = "HIGH_HRV"
        else:
            hrv_label = "TYPICAL_HRV"

        stress_score = {"CALM": 1.0, "NORMAL": 3.0, "STRESSED": 5.0}.get(stress, 3.0)
        arousal_score = round(1.0 + 4.0 * arousal, 1)
        if signal_quality == "GOOD":
            # Both inputs rise monotonically with activation/stress. WESAD is
            # weighted more strongly because it is the trained stress output.
            bio_state_score = round(0.7 * stress_score + 0.3 * arousal_score, 1)
            bio_state_label = (
                "VERY_CALM" if bio_state_score <= 1.5 else
                "CALM" if bio_state_score <= 2.5 else
                "MODERATE" if bio_state_score <= 3.5 else
                "ELEVATED" if bio_state_score <= 4.5 else "HIGH_STRESS"
            )
        else:
            # Zero is reserved for missing or unreliable camera physiology;
            # it is never interpreted as a calm-state estimate.
            bio_state_score = 0.0
            bio_state_label = "UNAVAILABLE"

        return {
            "stress": stress,
            "stress_score_5": stress_score,
            "bio_state_label": bio_state_label,
            "bio_state_score_5": bio_state_score,
            "arousal": {"label": arousal_label, "score_5": arousal_score},
            "valence": {"label": valence_label, "score_5": round(1.0 + 2.0 * (valence + 1.0), 1)},
            "heart_rate": {"label": heart_rate_label, "score_5": score_1_to_5(heart_rate, 50.0, 120.0)},
            # In this field, 5 means higher stress load / lower HRV.
            "hrv_stress_load": {
                "label": hrv_label,
                "score_5": score_1_to_5(rmssd, 10.0, 100.0, invert=True),
            },
        }

    async def _update_valence(self):
        while True:
            await asyncio.sleep(2)
            samples = self.microphone.recent_float_audio()
            if len(samples) >= INPUT_RATE:
                features = await asyncio.to_thread(extract_audio_features, samples, INPUT_RATE)
                self.current_state["valence"] = float(estimate_voice_valence(features)) if features.get("has_speech") else 0.0

    async def _receive_server(self, ws):
        async for raw in ws:
            content = json.loads(raw).get("serverContent", {})
            if content.get("interrupted"):
                self._model_speaking = False
                self.playback.clear()
                await self.hud.update({"llm_response": ""})
                continue
            # These are deliberately handled as two independent channels.
            # Never copy output transcription into `transcript` (VOICE), or
            # input transcription into `llm_response` (COMPANION).
            input_text = self._read_transcript(content, "inputTranscription")
            if input_text:
                await self.hud.update({"transcript": input_text})

            output_text = self._read_transcript(content, "outputTranscription")
            if output_text:
                await self.hud.update({"llm_response": output_text})
            for part in content.get("modelTurn", {}).get("parts", []):
                inline = part.get("inlineData") or {}
                if inline.get("data"):
                    speech_at = self.microphone.last_speech_at
                    if speech_at > self._last_response_latency_log:
                        print(f"[Live Agent] First audio latency: {time.monotonic() - speech_at:.2f}s")
                        self._last_response_latency_log = speech_at
                    self._model_speaking = True
                    self.playback.enqueue(base64.b64decode(inline["data"]))
            if content.get("turnComplete"):
                self._model_speaking = False

    @staticmethod
    def _read_transcript(content, field):
        """Read exactly one Gemini Live transcript channel.

        Gemini's raw WebSocket payload uses `inputTranscription` and
        `outputTranscription`. Do not fall back from one channel to another:
        that makes the user's text appear as the assistant response (or vice
        versa) when a partial payload is received.
        """
        value = content.get(field)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            text = value.get("text")
            return text.strip() if isinstance(text, str) else ""
        return ""


def main():
    try:
        asyncio.run(GeminiLiveEmotionAgent().run())
    except KeyboardInterrupt:
        print("\n[Live Agent] Stopped.")


if __name__ == "__main__":
    main()
