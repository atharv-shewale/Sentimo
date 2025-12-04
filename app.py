# app.py
import os
import time
import json
import logging
import random
import asyncio
from typing import Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# ----------------------------
# Try to import your classifier module
# ----------------------------
try:
    # Expected to provide: facemesh, get_features, classify_emotion, JOKES
    from emotion_classifier import facemesh, get_features, classify_emotion, JOKES
except Exception:
    facemesh = None
    JOKES = ["Why don't scientists trust atoms? Because they make up everything!"]

    def get_features(landmarks):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def classify_emotion(*args, **kwargs):
        return "neutral"

# ----------------------------
# Logging and app setup
# ----------------------------
logger = logging.getLogger("sentimo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

LAST_EMOTION_FILE = "/tmp/last_emotion.txt"


def save_last_emotion(emotion: str):
    try:
        with open(LAST_EMOTION_FILE, "w", encoding="utf-8") as f:
            f.write(f"{time.time()}|{emotion}")
    except Exception:
        logger.exception("Failed to save last emotion")


@app.get("/")
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/last")
async def last():
    try:
        if not os.path.exists(LAST_EMOTION_FILE):
            return {"last": None}
        with open(LAST_EMOTION_FILE, "r", encoding="utf-8") as f:
            line = f.read().strip()
        if not line:
            return {"last": None}
        ts, emotion = line.split("|", 1)
        return {"last": emotion, "timestamp": float(ts)}
    except Exception:
        logger.exception("Error reading last emotion")
        return Response(status_code=500, content="error")


# ----------------------------
# Helpers
# ----------------------------
def base64_b64decode(s: str) -> bytes:
    import base64
    if isinstance(s, str):
        s = s.encode("utf-8")
    return base64.b64decode(s)


def decode_frame_to_bgr(frame_data_b64: str):
    """
    Decode base64 JPEG string (optionally data URI) to BGR image (cv2 format).
    """
    if not frame_data_b64:
        return None
    if frame_data_b64.startswith("data:image"):
        frame_data_b64 = frame_data_b64.split(",", 1)[1]
    try:
        np_bytes = base64_b64decode(frame_data_b64)
        nparr = np.frombuffer(np_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        logger.exception("decode_frame_to_bgr failed")
        return None


def process_frame_and_classify(frame_bgr) -> Tuple[str, str]:
    """
    Downscale, convert to RGB, run mediapipe facemesh, and classify.
    Offload this to asyncio.to_thread to avoid blocking the event loop.
    Returns: (emotion, joke_or_None)
    """
    try:
        # Downscale for performance
        small = cv2.resize(frame_bgr, (320, 240))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if facemesh is None:
            return "neutral", None

        results = facemesh.process(rgb)
        if not results or not getattr(results, "multi_face_landmarks", None):
            return "No face", None

        lm = results.multi_face_landmarks[0].landmark
        features = get_features(lm)
        emotion = classify_emotion(*features)
        joke = random.choice(JOKES) if emotion == "sad" else None
        return emotion, joke
    except Exception:
        logger.exception("process_frame_and_classify failed")
        return "error", None


# ----------------------------
# WebSocket endpoint (supports binary blobs + JSON)
# ----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected (binary-capable)")

    try:
        while True:
            # Low-level receive: returns dict with 'type' and either 'text' or 'bytes'
            msg = await websocket.receive()
            mtype = msg.get("type")
            if mtype == "websocket.disconnect":
                logger.info("websocket.disconnect received")
                break

            # TEXT payload
            if "text" in msg and msg["text"] is not None:
                txt = msg["text"]
                logger.info("Received text payload len=%d", len(txt))
                try:
                    data = json.loads(txt)
                except Exception:
                    logger.exception("Failed to parse JSON text payload")
                    continue

                # heartbeat ping/pong
                if isinstance(data, dict) and data.get("type") == "ping":
                    try:
                        await websocket.send_json({"type": "pong"})
                        logger.info("Handled ping->pong")
                    except Exception:
                        logger.exception("Failed to send pong")
                    continue

                # if client still sends base64 JSON frames (optional)
                if isinstance(data, dict) and data.get("frame"):
                    logger.info("Received base64 frame; payload len=%d", len(data.get("frame")))
                    try:
                        frame = await asyncio.to_thread(decode_frame_to_bgr, data.get("frame"))
                        if frame is None:
                            await websocket.send_json({"emotion": "No face", "joke": None})
                            save_last_emotion("No face")
                            continue
                        emotion, joke = await asyncio.to_thread(process_frame_and_classify, frame)
                        await websocket.send_json({"emotion": emotion, "joke": joke})
                        save_last_emotion(emotion)
                        logger.info("Sent emotion=%s", emotion)
                    except Exception:
                        logger.exception("Error processing base64 JSON frame")
                        await websocket.send_json({"emotion": "error", "joke": None})
                    continue

            # BINARY payload
            if "bytes" in msg and msg["bytes"] is not None:
                b = msg["bytes"]
                logger.info("Received binary frame, size=%d", len(b))
                try:
                    nparr = np.frombuffer(b, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    logger.exception("Failed to decode binary frame")
                    await websocket.send_json({"emotion": "error", "joke": None})
                    continue

                if frame is None:
                    logger.warning("cv2.imdecode returned None")
                    await websocket.send_json({"emotion": "error", "joke": None})
                    continue

                try:
                    emotion, joke = await asyncio.to_thread(process_frame_and_classify, frame)
                    await websocket.send_json({"emotion": emotion, "joke": joke})
                    save_last_emotion(emotion)
                    logger.info("Sent emotion=%s", emotion)
                except Exception:
                    logger.exception("Error during processing binary frame")
                    await websocket.send_json({"emotion": "error", "joke": None})
                continue

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("Unhandled websocket error")
    finally:
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket connection closed (cleanup)")
