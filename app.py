# app.py
import base64
import random
import asyncio
import logging

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# import your existing classifier (must expose facemesh, get_features, classify_emotion, JOKES)
from emotion_classifier import facemesh, get_features, classify_emotion, JOKES

app = FastAPI()

# Logging
logger = logging.getLogger("sentimo")
logging.basicConfig(level=logging.INFO)

# Serve static files (sleep.mp3 etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


def decode_frame_to_bgr(frame_data_b64: str):
    """Decode base64 JPEG string to BGR image."""
    if frame_data_b64.startswith("data:image"):
        frame_data_b64 = frame_data_b64.split(",", 1)[1]

    np_bytes = base64.b64decode(frame_data_b64)
    nparr = np.frombuffer(np_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def process_frame_and_classify(frame_bgr):
    """
    Heavy CPU work: resize, run mediapipe, extract features, classify.
    Runs in a worker thread via asyncio.to_thread.
    """
    # Downscale to reduce CPU load
    small = cv2.resize(frame_bgr, (320, 240))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    results = facemesh.process(rgb)
    if not results.multi_face_landmarks:
        return "No face", None

    lm = results.multi_face_landmarks[0].landmark
    features = get_features(lm)
    emotion = classify_emotion(*features)
    joke = random.choice(JOKES) if emotion == "sad" else None
    return emotion, joke


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception:
                logger.exception("Error receiving websocket data")
                break

            frame_data = data.get("frame")
            if not frame_data:
                continue

            # Decode image (fast) then offload heavy work to thread
            try:
                frame = await asyncio.to_thread(decode_frame_to_bgr, frame_data)
                if frame is None:
                    logger.warning("Decoded frame is None")
                    continue
            except Exception:
                logger.exception("Frame decode error")
                continue

            # Offload mediapipe to thread so worker is responsive
            try:
                emotion, joke = await asyncio.to_thread(process_frame_and_classify, frame)
            except Exception:
                logger.exception("Error during mediapipe processing")
                emotion, joke = "error", None

            # Send JSON response
            try:
                await websocket.send_json({"emotion": emotion, "joke": joke})
            except Exception:
                logger.exception("Failed to send websocket message")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnect outside receive loop")
    except Exception:
        logger.exception("WebSocket main loop error")
