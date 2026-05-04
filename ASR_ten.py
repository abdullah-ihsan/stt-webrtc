import asyncio
import json
import logging
import os
import av
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from networkx import config

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("webrtc-transcription")

app = FastAPI(title="WebRTC Multi-Provider Transcription")
pcs: set[RTCPeerConnection] = set()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

KEYS = {
    "speechmatics": os.getenv("SPEECHMATICS_API_KEY"),
    "assemblyai": os.getenv("ASSEMBLYAI_API_KEY"),
    "deepgram": os.getenv("DEEPGRAM_API_KEY"),
}

PCM_SAMPLE_RATE = 16000
MSG_PARTIAL = "PARTIAL"
MSG_FINAL = "FINAL"

# AssemblyAI
AAI_MODEL         = "u3-rt-pro"
AAI_EOT_THRESHOLD = 0.4     # end-of-turn confidence; lower = faster turn detection

AAI_WS_URL = (
    f"wss://streaming.assemblyai.com/v3/ws"
    f"?speech_model={AAI_MODEL}"
    f"&sample_rate={PCM_SAMPLE_RATE}"
    f"&encoding=pcm_s16le"
    f"&format_turns=true"
    f"&end_of_turn_confidence_threshold={AAI_EOT_THRESHOLD}"
)


# ── Base Handler with Clean Lifecycle ────────────────────────────────────────

class BaseTranscriptionHandler(ABC):
    def __init__(self, track, data_channel):
        self.track = track
        self.dc = data_channel
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=PCM_SAMPLE_RATE)
        self._task: asyncio.Task | None = None
        self._closed = False

        # Push-to-talk: accumulate all final transcript segments for this turn
        self._turn_finals: list[str] = []
        self._stop_requested = asyncio.Event()

    def handle_client_message(self, raw: str):
        """Called when client sends a message over the data channel (e.g. stop signal)."""
        try:
            msg = json.loads(raw)
            if msg.get("type") == "stop":
                log.info("Client requested turn stop — flushing turn")
                self._stop_requested.set()
        except Exception:
            pass

    async def flush_turn(self):
        """Emit the complete turn text as a single 'turn_complete' message."""
        SENTENCE_END = {".", "!", "?"}
        parts = [s.strip() for s in self._turn_finals if s.strip()]
        joined = []
        for i, part in enumerate(parts):
            if i > 0 and joined and joined[-1][-1] not in SENTENCE_END:
                joined[-1] += "."
            joined.append(part)
        full_text = " ".join(joined)
        self._turn_finals.clear()
        if full_text and self.dc and self.dc.readyState == "open":
            try:
                payload = json.dumps({"type": "turn_complete", "text": full_text})
                self.dc.send(payload)
                log.info(f"Turn complete: {full_text[:80]}...")
            except Exception as e:
                log.warning(f"DataChannel send failed on flush: {e}")

    async def send_text(self, prefix: str, text: str):
        if not text or not text.strip() or self._closed:
            return

        # Accumulate finals into turn buffer (don't emit partials to LLM)
        if prefix == MSG_FINAL:
            self._turn_finals.append(text.strip())

        if self.dc and self.dc.readyState == "open":
            try:
                payload = json.dumps({"type": prefix.lower(), "text": text.strip()})
                self.dc.send(payload)
            except Exception as e:
                log.warning(f"DataChannel send failed: {e}")

    async def get_audio_chunks(self):
        while not self._closed:
            try:
                frame = await self.track.recv()
                for resampled in self.resampler.resample(frame):
                    yield resampled.to_ndarray().tobytes()
            except Exception:
                break

    @abstractmethod
    async def run(self):
        pass

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run_wrapper())

    async def _run_wrapper(self):
        try:
            # Run transcription and wait for stop signal concurrently
            run_task = asyncio.create_task(self.run())
            stop_task = asyncio.create_task(self._stop_requested.wait())
            done, pending = await asyncio.wait(
                [run_task, stop_task], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        except asyncio.CancelledError:
            log.debug("Handler task cancelled")
        except Exception as e:
            log.error(f"Handler error: {e}", exc_info=True)
        finally:
            # Flush whatever was accumulated in this turn before closing
            await self.flush_turn()
            await self.close()

    async def close(self):
        if self._closed:
            return
        self._closed = True

        log.info("Closing transcription handler")

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Cleanup resampler
        try:
            if hasattr(self.resampler, "close"):
                self.resampler.close()
        except Exception:
            pass

        # Close data channel
        if self.dc and self.dc.readyState in ("open", "connecting"):
            try:
                self.dc.close()
            except Exception:
                pass


# ── Provider Handlers ────────────────────────────────────────────────────────

class SpeechmaticsHandler(BaseTranscriptionHandler):
    async def run(self):
        from speechmatics.rt import AsyncClient, AudioEncoding, AudioFormat, ServerMessageType, TranscriptionConfig

        async with AsyncClient(api_key=KEYS["speechmatics"]) as client:
            @client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
            def on_partial(msg):
                asyncio.create_task(self.send_text(MSG_PARTIAL, msg["metadata"]["transcript"]))

            @client.on(ServerMessageType.ADD_TRANSCRIPT)
            def on_final(msg):
                asyncio.create_task(self.send_text(MSG_FINAL, msg["metadata"]["transcript"]))

            await client.start_session(
                transcription_config=TranscriptionConfig(language="en", enable_partials=True),
                audio_format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE)
            )

            async for chunk in self.get_audio_chunks():
                if self._closed: break
                await client.send_audio(chunk)


class AssemblyAIHandler(BaseTranscriptionHandler):
    async def run(self):
        import websockets

        async with websockets.connect(
            AAI_WS_URL,
            additional_headers={"Authorization": KEYS["assemblyai"]}
        ) as ws:

            async def receiver():
                try:
                    async for msg in ws:
                        if self._closed: break
                        data = json.loads(msg)
                        if data.get("type") == "Turn":
                            is_final = data.get("end_of_turn", False)
                            prefix = MSG_FINAL if is_final else MSG_PARTIAL
                            await self.send_text(prefix, data.get("transcript", ""))
                        elif data.get("type") == "Error":
                            log.error(f"AssemblyAI Error: {data.get('error')}")
                except Exception as e:
                    if not self._closed:
                        log.error(f"AssemblyAI receiver failed: {e}")

            recv_task = asyncio.create_task(receiver())
            buffer = bytearray()
            MIN_CHUNK_SIZE = 2000

            try:
                async for chunk in self.get_audio_chunks():
                    if self._closed: break
                    buffer.extend(chunk)
                    if len(buffer) >= MIN_CHUNK_SIZE:
                        await ws.send(bytes(buffer))
                        buffer.clear()
            finally:
                if buffer and not self._closed:
                    await ws.send(bytes(buffer))
                await ws.send(json.dumps({"type": "Terminate"}))
                recv_task.cancel()
                await asyncio.gather(recv_task, return_exceptions=True)


class DeepgramHandler(BaseTranscriptionHandler):
    async def run(self):
        from deepgram import DeepgramClient
        from deepgram.core.events import EventType
        import threading

        loop = asyncio.get_running_loop()
        client = DeepgramClient(api_key=KEYS["deepgram"])

        with client.listen.v1.connect(
            model="nova-3", encoding="linear16", sample_rate=PCM_SAMPLE_RATE
        ) as conn:

            def on_message(message, **kwargs):
                try:
                    text = message.channel.alternatives[0].transcript
                    if not text:
                        return
                    is_final = getattr(message, "is_final", False)
                    asyncio.run_coroutine_threadsafe(
                        self.send_text(MSG_FINAL if is_final else MSG_PARTIAL, text), loop
                    )
                except Exception as e:
                    log.warning(f"Deepgram parse error: {e}")

            conn.on(EventType.MESSAGE, on_message)
            threading.Thread(target=conn.start_listening, daemon=True).start()

            try:
                async for chunk in self.get_audio_chunks():
                    if self._closed: break
                    conn.send_media(chunk)
            finally:
                try:
                    conn.send_close_stream()
                except Exception:
                    pass


# ── Handler Registry ─────────────────────────────────────────────────────────

HANDLERS = {
    "speechmatics": SpeechmaticsHandler,
    "assemblyai": AssemblyAIHandler,
    "deepgram": DeepgramHandler,
}


# ── Cleanup Utility ──────────────────────────────────────────────────────────

async def cleanup_pc(pc: RTCPeerConnection):
    handler = getattr(pc, "handler", None)
    if handler:
        await handler.close()

    try:
        await pc.close()
    except Exception as e:
        log.warning(f"Error closing PC: {e}")

    pcs.discard(pc)


# ── WebRTC Routes ────────────────────────────────────────────────────────────

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    provider = params.get("provider", "assemblyai").lower()

    if provider not in HANDLERS:
        return JSONResponse({"error": "Unsupported provider"}, status_code=400)

    config = RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(
            urls="turn:free.expressturn.com:3478",
            username=os.getenv("EXPRESS_TURN_USERNAME"),
            credential=os.getenv("EXPRESS_TURN_CREDENTIAL")
        )
    ])

    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ("failed", "closed"):
            await cleanup_pc(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState in ("failed", "closed"):
            await cleanup_pc(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind != "audio":
            return

        @pc.on("datachannel")
        def on_datachannel(dc):
            handler = HANDLERS[provider](track, dc)
            pc.handler = handler          # Attach for cleanup
            asyncio.create_task(handler.start())

            @dc.on("message")
            def on_dc_message(msg):
                handler.handle_client_message(msg)

            @track.on("ended")
            async def on_track_ended():
                await cleanup_pc(pc)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


@app.get("/")
async def index():
    return HTMLResponse(open("index.html").read())

@app.on_event("shutdown")
async def on_shutdown():
    log.info(f"Shutting down: closing {len(pcs)} connections")
    for pc in list(pcs):
        await cleanup_pc(pc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9800)