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

# ----------------------------
# IMPORT / EMBED YOUR CLASSIFIER
# ----------------------------
# If you already have emotion_classifier.py exposing facemesh, get_features,
# classify_emotion and JOKES, you can import them instead:
#
# from emotion_classifier import facemesh, get_features, classify_emotion, JOKES
#
# For simplicity this file expects that you have an emotion_classifier.py
# with the required objects. If not, move the functions below into a separate
# module or paste your classifier here.
#
try:
    from emotion_classifier import facemesh, get_features, classify_emotion, JOKES
except Exception:
    # If import fails, attempt to create a lightweight fallback to avoid import errors
    # (Real classification will likely not work with this fallback. Replace with your module.)
    facemesh = None
    JOKES = ["Why don't scientists trust atoms? Because they make up everything!"]

    def get_features(landmarks):
        # Minimal dummy values to avoid crashes; replace with your real function.
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def classify_emotion(*args, **kwargs):
        return "neutral"

# ----------------------------
# APPLICATION & LOGGING SETUP
# ----------------------------
logger = logging.getLogger("sentimo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# last emotion persistence for debugging
LAST_EMOTION_FILE = "/tmp/last_emotion.txt"


def save_last_emotion(emotion: str):
    try:
        with open(LAST_EMOTION_FILE, "w", encoding="utf-8") as f:
            f.write(f"{time.time()}|{emotion}")
    except Exception:
        logger.exception("Failed to save last emotion")


@app.get("/")
async def index():
    # Serve index.html from repo root
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
# Helper functions
# ----------------------------
def decode_frame_to_bgr(frame_data_b64: str):
    """
    Decode base64 JPEG string to BGR image (cv2 format).
    Accepts either a raw base64 image string or data URI ("data:image/...,base64,...")
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


def base64_b64decode(s: str) -> bytes:
    # Local wrapper to avoid importing base64 everywhere, and to safely handle str/bytes
    import base64
    if isinstance(s, str):
        s = s.encode("utf-8")
    return base64.b64decode(s)


def process_frame_and_classify(frame_bgr) -> Tuple[str, str]:
    """
    Downscale, convert to RGB, run mediapipe (facemesh), extract features and classify.
    This function is CPU-bound and must be called inside asyncio.to_thread.
    """
    try:
        # small resize for performance
        small = cv2.resize(frame_bgr, (320, 240))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if facemesh is None:
            # fallback classifier (if real facemesh not available)
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
# WebSocket endpoint (low-level receive; supports JSON/base64 + binary Blobs)
# ----------------------------
from fastapi import WebSocket
from starlette.websockets import WebSocketState

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected (low-level receive)")

    try:
        while True:
            # Low-level receive: catches text or bytes payloads
            msg = await websocket.receive()
            # msg typically contains: {"type":"websocket.receive","text": "..."} or {"type":"websocket.receive","bytes": b'...'}
            mtype = msg.get("type")
            if mtype == "websocket.disconnect":
                logger.info("Received websocket.disconnect")
                break

            # TEXT payload (JSON)
            if "text" in msg and msg["text"] is not None:
                txt = msg["text"]
                logger.info("Received text payload len=%d", len(txt))
                # try parse JSON
                try:
                    data = json.loads(txt)
                except Exception:
                    logger.exception("Failed to parse JSON text payload")
                    # skip if invalid
                    continue

                # handle ping JSON
                if isinstance(data, dict) and data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    logger.info("Handled ping->pong")
                    continue

                # handle base64-framed JSON
                if isinstance(data, dict) and data.get("frame"):
                    logger.info("Received base64 frame; payload len=%d", len(data.get("frame")))
                    # decode -> process in a thread
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

            # BINARY payload (bytes)
            if "bytes" in msg and msg["bytes"] is not None:
                b = msg["bytes"]
                logger.info("Received binary frame, size=%d", len(b))
                # decode image bytes to cv2 frame
                try:
                    nparr = np.frombuffer(b, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    logger.exception("Failed to decode binary frame buffer")
                    await websocket.send_json({"emotion": "error", "joke": None})
                    continue

                if frame is None:
                    logger.warning("cv2.imdecode returned None for binary frame")
                    await websocket.send_json({"emotion": "error", "joke": None})
                    continue

                # offload heavy processing
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
