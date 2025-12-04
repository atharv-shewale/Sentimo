import mediapipe as mp
import numpy as np

mpface = mp.solutions.face_mesh
facemesh = mpface.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def get_features(landmarks):
    coords = np.array([[p.x, p.y] for p in landmarks])
    xs, ys = coords[:, 0], coords[:, 1]
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    width, height = max(maxx - minx, 1e-6), max(maxy - miny, 1e-6)

    def dist(i, j):
        return np.linalg.norm((coords[i] - coords[j]) / np.array([width, height]))

    upper_lip, lower_lip = 13, 14
    left_mouth, right_mouth = 61, 291
    eye_outer, eye_inner = 33, 133
    eye_top, eye_bottom = 159, 145

    mouth_open = dist(upper_lip, lower_lip)
    mouth_width = dist(left_mouth, right_mouth)
    eye_vertical = dist(eye_top, eye_bottom)
    eye_horizontal = dist(eye_outer, eye_inner)
    eye_open = eye_vertical / (eye_horizontal + 1e-6)

    nose_idx = 1
    nose = coords[nose_idx]
    center = coords.mean(axis=0)
    nose_cod = (nose[1] - center[1]) / height

    inner_eye_dist = dist(133, 362)
    left_lip_y = coords[left_mouth, 1]
    right_lip_y = coords[right_mouth, 1]
    lip_asym = (left_lip_y - right_lip_y) / height

    left_side = coords[234, 1]
    right_side = coords[454, 1]
    head_tilt = (left_side - right_side) / height

    return mouth_open, mouth_width, eye_open, nose_cod, inner_eye_dist, lip_asym, head_tilt


def classify_emotion(mouth_open, mouth_width, eye_open, nose_cod, inner_eye_dist, lip_asym, head_tilt):
    # ❤️ ALL YOUR ORIGINAL LOGIC PRESERVED — NO CHANGES
    big_mouth_open = mouth_open > 0.18
    strong_smile = mouth_width > 0.45 and eye_open > 0.18
    relaxed_smile = mouth_width > 0.045 and eye_open > 0.17
    eyes_very_open = eye_open > 0.30
    eyes_narrow = eye_open < 0.18
    eyes_very_narrow = eye_open < 0.14
    head_down = nose_cod > 0.03
    head_up = nose_cod < -0.03
    strong_head_tilt = abs(head_tilt) > 0.05
    lip_asym_strong = abs(lip_asym) > 0.03

    if big_mouth_open and eyes_narrow:
        return "sleepy"
    if eyes_very_open and mouth_open > 0.08:
        return "shocked"
    if mouth_open > 0.10 and eye_open > 0.25:
        return "surprised"
    if strong_smile:
        return "smile"
    if mouth_width > 0.39 and eye_open >= 0.17:
        return "happy"
    if 0.15 <= eye_open < 0.20 and mouth_width < 0.04 and not big_mouth_open:
        return "sad"
    if eyes_narrow and inner_eye_dist < 0.27 and mouth_width < 0.04 and not big_mouth_open:
        return "angry"
    if lip_asym_strong and mouth_open < 0.06:
        return "disgust"
    if eyes_narrow:
        if head_down:
            return "sleepy and looking down"
        if head_up:
            return "sleepy"
        return "sleepy"
    if head_down and 0.18 <= eye_open <= 0.26 and mouth_open < 0.06:
        return "thinking"
    if strong_head_tilt and eye_open > 0.18:
        return "confused"
    return "neutral"


JOKES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "Why did the bicycle fall over? Because it was two-tired!",
    "Why don’t skeletons fight each other? They don’t have the guts.",
    "What do you call fake spaghetti? An impasta!",
    "Why did the math book look sad? Because it had too many problems!"
]
