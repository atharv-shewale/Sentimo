from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from emotion_classifier import facemesh, get_features, classify_emotion, JOKES
import cv2, numpy as np, base64, random, os

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            data = await websocket.receive_json()
            frame_data = data.get("frame")

            if not frame_data:
                continue

            # Strip header if present
            if frame_data.startswith("data:image"):
                frame_data = frame_data.split(",", 1)[1]

            # Decode base64 → numpy → image
            try:
                np_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(np_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as decode_error:
                print("Frame decode error:", decode_error)
                continue

            # Resize + convert to RGB
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = facemesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                features = get_features(lm)
                emotion = classify_emotion(*features)
            else:
                emotion = "No face"

            # Jokes only when sad
            joke = random.choice(JOKES) if emotion == "sad" else None

            # Send emotion (frontend displays webcam itself!)
            await websocket.send_json({
                "emotion": emotion,
                "joke": joke
            })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("WebSocket Error:", e)
