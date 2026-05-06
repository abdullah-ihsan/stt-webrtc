import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("ws-transcription")

app = FastAPI(title="WebSocket Transcription")

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
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self._task: asyncio.Task | None = None
        self._closed = False
        self._turn_finals: list[str] = []
        self._stop_requested = asyncio.Event()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)

    def handle_client_message(self, raw: str):
        try:
            msg = json.loads(raw)
            if msg.get("type") == "stop":
                self._stop_requested.set()
        except:
            pass


    async def flush_turn(self):
        if self._closed:
            return
        SENTENCE_END = {".", "!", "?"}
        parts = [s.strip() for s in self._turn_finals if s.strip()]
        joined = []
        for i, part in enumerate(parts):
            if i > 0 and joined and joined[-1][-1] not in SENTENCE_END:
                # joined[-1] += "."
                pass
            joined.append(part)
        full_text = " ".join(joined).strip()
        self._turn_finals.clear()

        if full_text:
            try:
                payload = json.dumps({"type": "turn_complete", "text": full_text})
                await self.ws.send_text(payload)
                log.info(f"✅ Turn complete sent: {full_text[:80]}...")
            except:
                self._closed = True

    async def send_text(self, prefix: str, text: str):
        if not text or not text.strip() or self._closed:
            return

        if prefix == MSG_FINAL:
            self._turn_finals.append(text.strip())

        try:
            payload = json.dumps({"type": prefix.lower(), "text": text.strip()})
            await self.ws.send_text(payload)
        except Exception as e:
            log.warning(f"WS send failed: {e}")
            self._closed = True


    async def get_audio_chunks(self) -> AsyncGenerator[bytes, None]:
        while not self._closed:
            try:
                chunk = await self._audio_queue.get()
                if chunk is None:
                    break
                yield chunk
            except:
                break

    async def put_audio(self, data: bytes):
        if not self._closed:
            try:
                await self._audio_queue.put(data)
            except asyncio.QueueFull:
                pass  # drop old chunks

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


# ── Provider Handlers (Mostly Unchanged) ─────────────────────────────────────

class SpeechmaticsHandler(BaseTranscriptionHandler):
    async def run(self):
        from speechmatics.rt import AsyncClient, AudioEncoding, AudioFormat, ServerMessageType, TranscriptionConfig

        async with AsyncClient(api_key=KEYS["speechmatics"]) as client:
            @client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
            def on_partial(msg):
                partial_text = msg["metadata"]["transcript"].strip()
                full_text = " ".join(self._turn_finals + [partial_text]).strip()
                asyncio.create_task(self.send_text(MSG_PARTIAL, full_text))

            @client.on(ServerMessageType.ADD_TRANSCRIPT)
            def on_final(msg):
                text = msg["metadata"]["transcript"].strip()
                if text:
                    self._turn_finals.append(text)
                    full_text = " ".join(self._turn_finals).strip()
                    asyncio.create_task(self.send_text(MSG_PARTIAL, full_text))

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
            MIN_CHUNK_SIZE = PCM_SAMPLE_RATE * 2 // 2  # 0.5 seconds of audio (16-bit mono)

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


# ── WebSocket Endpoint (Robust) ─────────────────────────────────────────────
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    log.info("WebSocket client connected")

    provider = "assemblyai"  # default provider
    handler = None

    try:
        # Get provider from first message
        try:
            first_msg = await websocket.receive_text()
            data = json.loads(first_msg)
            provider = data.get("provider", "assemblyai").lower()
        except:
            pass
        if provider not in HANDLERS:
            provider = "assemblyai"

        log.info(f"WebSocket using provider: {provider}")

        # Create handler
        handler = HANDLERS[provider](websocket)   # ← This is the correct call now

        await handler.start()

        while not handler._closed:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=0.8)
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        await handler.put_audio(message["bytes"])
                    elif "text" in message:
                        handler.handle_client_message(message["text"])
            except asyncio.TimeoutError:
                continue
            except (WebSocketDisconnect, Exception):
                break

    except WebSocketDisconnect:
        log.info("Client disconnected normally")
    except Exception as e:
        log.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if handler:
            await handler.close()
        log.info("WebSocket connection closed")


@app.get("/")
async def index():
    return HTMLResponse(open("index.html").read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9800, ssl_keyfile="192.168.100.2-key.pem", ssl_certfile="192.168.100.2.pem")