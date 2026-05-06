import asyncio
import contextlib
import json
import logging
import os
import sys
import uuid
from abc import ABC, abstractmethod
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from livekit import api, rtc

from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("carebuddy-ws")

app = FastAPI(title="CareBuddy Live")

# ====================== MAYA IMPORT ======================
sys.path.append(".")
from llm_layer import build_navigator, initial_greeting

graph = build_navigator()

initial_message = initial_greeting("This is the first conversation.")
active_sessions = {}
livekit_sessions = {}

# ====================== TRANSCRIPTION CONFIG ======================
KEYS = {
    "speechmatics": os.getenv("SPEECHMATICS_API_KEY"),
    "assemblyai": os.getenv("ASSEMBLYAI_API_KEY"),
    "deepgram": os.getenv("DEEPGRAM_API_KEY"),
}

PCM_SAMPLE_RATE = 16000
MSG_PARTIAL = "PARTIAL"
MSG_FINAL = "FINAL"
LIVEKIT_EVENTS_TOPIC = "carebuddy.events"
LIVEKIT_CONTROL_TOPIC = "carebuddy.control"

AAI_MODEL = "u3-rt-pro"
AAI_EOT_THRESHOLD = 0.4
AAI_WS_URL = (
    f"wss://streaming.assemblyai.com/v3/ws"
    f"?speech_model={AAI_MODEL}"
    f"&sample_rate={PCM_SAMPLE_RATE}"
    f"&encoding=pcm_s16le"
    f"&format_turns=true"
    f"&end_of_turn_confidence_threshold={AAI_EOT_THRESHOLD}"
)

# ====================== BASE HANDLER (with Maya) ======================
class BaseTranscriptionHandler(ABC):
    def __init__(self, transport):
        self.ws = transport
        self._task: asyncio.Task | None = None
        self._closed = False
        self._turn_finals: list[str] = []
        self._latest_partial: str = ""
        self._stop_requested = asyncio.Event()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self.session_id = id(transport)
        self.config = {"configurable": {"thread_id": f"carebuddy_{self.session_id}"}}

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
        if self._latest_partial.strip():
            parts.append(self._latest_partial.strip())
            self._latest_partial = ""

        joined = []
        for i, part in enumerate(parts):
            if i > 0 and joined and joined[-1][-1] not in SENTENCE_END:
                # joined[-1] += "."
                pass
            joined.append(part)

        full_text = " ".join(joined).strip()
        self._turn_finals.clear()

        if not full_text:
            return

        try:
            # Send turn complete to frontend
            await self.ws.send_text(json.dumps({"type": "turn_complete", "text": full_text}))
            log.info(f"✅ Turn complete sent: {full_text[:80]}...")

            # === CALL MAYA ===
            session = active_sessions.get(self.session_id)
            if session:
                result = graph.invoke({
                    "messages": [HumanMessage(content=full_text)],
                    "known_info": session.get("known_info", {"topic_completion": {}}),
                    "current_topic": session.get("current_topic", "Initial rapport & check-in"),
                    "emotional_tone": session.get("emotional_tone", "neutral"),
                    "visited_topics": session.get("visited_topics", [])
                }, config=self.config)

                ai_reply = result['messages'][-1].content

                # Update session
                session.setdefault("messages", []).extend([
                    HumanMessage(content=full_text),
                    AIMessage(content=ai_reply)
                ])
                session["known_info"] = result.get("known_info", session.get("known_info"))
                session["current_topic"] = result.get("current_topic")
                session["emotional_tone"] = result.get("emotional_tone")
                session["visited_topics"] = result.get("visited_topics", [])

                # Send AI response
                await self.ws.send_text(json.dumps({
                    "type": "ai_response",
                    "text": ai_reply
                }))
                log.info(f"🤖 Maya replied: {ai_reply[:80]}...")

        except Exception as e:
            log.error(f"Error in flush_turn + Maya: {e}", exc_info=True)

    async def send_text(self, prefix: str, text: str):
        if not text or not text.strip() or self._closed:
            return
        if prefix == MSG_FINAL:
            self._turn_finals.append(text.strip())
            self._latest_partial = ""
        elif prefix == MSG_PARTIAL:
            self._latest_partial = text.strip()

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
                pass

    @abstractmethod
    async def run(self):
        pass

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run_wrapper())


    async def _run_wrapper(self):
        try:
            run_task = asyncio.create_task(self.run())
            stop_task = asyncio.create_task(self._stop_requested.wait())
            done, pending = await asyncio.wait([run_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            log.error(f"Handler error: {e}", exc_info=True)
        finally:
            await self.flush_turn()
            await self.close()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self._task and not self._task.done() and self._task is not asyncio.current_task():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# ====================== PROVIDER HANDLERS ======================
class SpeechmaticsHandler(BaseTranscriptionHandler):
    async def run(self):
        from speechmatics.rt import AsyncClient, AudioEncoding, AudioFormat, ServerMessageType, TranscriptionConfig
        async with AsyncClient(api_key=KEYS["speechmatics"]) as client:
            @client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
            def on_partial(msg):
                partial_text = msg["metadata"]["transcript"].strip()
                asyncio.create_task(self.send_text(MSG_PARTIAL, partial_text))

            @client.on(ServerMessageType.ADD_TRANSCRIPT)
            def on_final(msg):
                text = msg["metadata"]["transcript"].strip()
                if text:
                    asyncio.create_task(self.send_text(MSG_FINAL, text))

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
            model="nova-3-medical", encoding="linear16", sample_rate=PCM_SAMPLE_RATE
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
                    log.warning(f"Deepgram error: {e}")

            conn.on(EventType.MESSAGE, on_message)
            threading.Thread(target=conn.start_listening, daemon=True).start()

            try:
                async for chunk in self.get_audio_chunks():
                    if self._closed: break
                    conn.send_media(chunk)
            finally:
                try:
                    conn.send_close_stream()
                except:
                    pass


HANDLERS = {
    "speechmatics": SpeechmaticsHandler,
    "assemblyai": AssemblyAIHandler,
    "deepgram": DeepgramHandler,
}


def new_session_state():
    return {
        "messages": [AIMessage(content=initial_message)],
        "known_info": {"topic_completion": {}},
        "current_topic": "Initial rapport & check-in",
        "emotional_tone": "neutral",
        "visited_topics": []
    }


def create_livekit_token(
    *,
    identity: str,
    room_name: str,
    display_name: str,
    can_publish: bool,
    can_subscribe: bool,
) -> str:
    livekit_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not livekit_url or not api_key or not api_secret:
        raise HTTPException(
            status_code=500,
            detail="LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must be set.",
        )

    return (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(display_name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )


class LiveKitDataTransport:
    def __init__(self, room: rtc.Room):
        self.room = room
        self.destination_identity: str | None = None

    async def send_text(self, payload: str):
        destinations = [self.destination_identity] if self.destination_identity else []
        await self.room.local_participant.publish_data(
            payload,
            reliable=True,
            destination_identities=destinations,
            topic=LIVEKIT_EVENTS_TOPIC,
        )


class LiveKitSessionBridge:
    def __init__(self, room_name: str, provider: str):
        self.room_name = room_name
        self.provider = provider if provider in HANDLERS else "speechmatics"
        self.identity = f"carebuddy-server-{uuid.uuid4().hex[:8]}"
        self.room = rtc.Room()
        self.transport = LiveKitDataTransport(self.room)
        self.session_id = id(self.transport)
        self.handler: BaseTranscriptionHandler | None = None
        self.turn_active = False
        self.closed = False
        self.initial_sent = False
        self.audio_tasks: dict[str, asyncio.Task] = {}

        active_sessions[self.session_id] = new_session_state()
        self._register_room_events()

    def _register_room_events(self):
        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            if participant.identity != self.identity:
                self.transport.destination_identity = participant.identity
                asyncio.create_task(self._send_initial_message())
                log.info("LiveKit participant connected: %s", participant.identity)

        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            if participant.identity == self.transport.destination_identity:
                asyncio.create_task(self.close())

        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            if participant.identity == self.identity:
                return
            self.transport.destination_identity = participant.identity
            task = asyncio.create_task(self._consume_audio(track, publication.sid))
            self.audio_tasks[publication.sid] = task
            task.add_done_callback(lambda _task: self.audio_tasks.pop(publication.sid, None))
            asyncio.create_task(self._send_initial_message())
            log.info("Subscribed to LiveKit audio track %s from %s", publication.sid, participant.identity)

        @self.room.on("data_received")
        def on_data_received(packet: rtc.DataPacket):
            if packet.topic and packet.topic != LIVEKIT_CONTROL_TOPIC:
                return
            if packet.participant:
                self.transport.destination_identity = packet.participant.identity
            try:
                message = json.loads(packet.data.decode("utf-8"))
            except Exception:
                log.warning("Ignoring invalid LiveKit control packet")
                return

            msg_type = message.get("type")
            if msg_type == "start":
                asyncio.create_task(self.start_turn())
            elif msg_type == "stop":
                asyncio.create_task(self.stop_turn())

    async def connect(self):
        livekit_url = os.getenv("LIVEKIT_URL")
        token = create_livekit_token(
            identity=self.identity,
            room_name=self.room_name,
            display_name="CareBuddy Server",
            can_publish=False,
            can_subscribe=True,
        )
        await self.room.connect(livekit_url, token)
        log.info("CareBuddy LiveKit bridge connected to room %s", self.room_name)

    async def _send_initial_message(self):
        if self.initial_sent or not self.transport.destination_identity:
            return
        self.initial_sent = True
        await self.transport.send_text(json.dumps({"type": "ai_response", "text": initial_message}))

    async def start_turn(self):
        if self.closed:
            return
        self.turn_active = True
        if self.handler and not self.handler._closed:
            return

        self.handler = HANDLERS[self.provider](self.transport)
        await self.handler.start()
        log.info("Started LiveKit transcription turn using provider: %s", self.provider)

    async def stop_turn(self):
        self.turn_active = False
        if not self.handler or self.handler._closed:
            return
        self.handler.handle_client_message(json.dumps({"type": "stop"}))

    async def _consume_audio(self, track: rtc.Track, publication_sid: str):
        stream = rtc.AudioStream(track, sample_rate=PCM_SAMPLE_RATE, num_channels=1)
        try:
            async for event in stream:
                if self.closed:
                    break
                if not self.turn_active or not self.handler or self.handler._closed:
                    continue
                await self.handler.put_audio(event.frame.data.tobytes())
        except Exception as e:
            if not self.closed:
                log.error("LiveKit audio stream failed for %s: %s", publication_sid, e, exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.turn_active = False
        for task in list(self.audio_tasks.values()):
            task.cancel()
        if self.audio_tasks:
            await asyncio.gather(*self.audio_tasks.values(), return_exceptions=True)
        if self.handler:
            await self.handler.close()
        active_sessions.pop(self.session_id, None)
        livekit_sessions.pop(self.room_name, None)
        if self.room.isconnected():
            await self.room.disconnect()
        log.info("Closed LiveKit bridge for room %s", self.room_name)


@app.get("/livekit/session")
async def livekit_session(provider: str = Query("speechmatics")):
    provider = provider.lower()
    if provider not in HANDLERS:
        provider = "speechmatics"

    room_name = f"carebuddy-{uuid.uuid4().hex[:12]}"
    participant_identity = f"patient-{uuid.uuid4().hex[:8]}"
    bridge = LiveKitSessionBridge(room_name, provider)

    try:
        await bridge.connect()
    except Exception as e:
        active_sessions.pop(bridge.session_id, None)
        log.error("Failed to start LiveKit bridge: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to connect LiveKit bridge: {e}")

    livekit_sessions[room_name] = bridge
    token = create_livekit_token(
        identity=participant_identity,
        room_name=room_name,
        display_name="Patient",
        can_publish=True,
        can_subscribe=True,
    )

    return {
        "url": os.getenv("LIVEKIT_URL"),
        "token": token,
        "room": room_name,
        "identity": participant_identity,
        "provider": provider,
    }


# ====================== WEBSOCKET ENDPOINT ======================
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    log.info("Client connected")

    handler = None
    session_id = id(websocket)

    # Initialize session
    active_sessions[session_id] = new_session_state()

    try:
        # Get provider
        try:
            first_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            data = json.loads(first_msg)
            provider = data.get("provider", "speechmatics").lower()
        except:
            provider = "speechmatics"

        if provider not in HANDLERS:
            provider = "speechmatics"

        log.info(f"Using provider: {provider}")

        handler = HANDLERS[provider](websocket)
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
        if session_id in active_sessions:
            active_sessions.pop(session_id, None)
        if handler:
            await handler.close()


@app.get("/")
async def index():
    return HTMLResponse(open("index.html").read())


if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=9800, ssl_keyfile="192.168.100.2-key.pem", ssl_certfile="192.168.100.2.pem")
    uvicorn.run(app, host="0.0.0.0", port=9800)