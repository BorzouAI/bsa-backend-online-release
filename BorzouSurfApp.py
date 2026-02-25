# =========================
# BorzouSurfApp.py
# Backend: Surf Analyzer
# =========================

import os
import hashlib
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import tempfile

app = FastAPI()

# --- CORS for Flutter/Web ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# --- Scoring config (simple, as you had) ---
BASE_POINTS = {
    "Bottom Turn": 2.0,
    "Top Turn": 2.4,
    "Cutback": 3.0,
    "Pumping": 0.5,
}
DECAY_FACTOR = 0.5  # decay after 3rd repeat


def wave_score(maneuver_log):
    total_score = 0.0
    counted = {}
    for m in maneuver_log:
        counted[m] = counted.get(m, 0) + 1
    for maneuver, count in counted.items():
        base = BASE_POINTS.get(maneuver, 0.0)
        for i in range(count):
            decay_multiplier = 1.0 if i < 3 else (DECAY_FACTOR ** (i - 2))
            total_score += base * decay_multiplier
    return min(round(total_score, 2), 9.99)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_point(lms, idx):
    i = idx.value if hasattr(idx, "value") else int(idx)
    return (lms[i].x, lms[i].y)


def detect_pumping(buf):
    if len(buf) < 10:
        return False
    y_vals = [p[1] for p in buf]
    diffs = np.diff(y_vals)
    if len(diffs) == 0:
        return False
    zero_crossings = np.sum(np.diff(np.sign(diffs)) != 0)
    avg_disp = np.mean(np.abs(diffs))
    horiz_disp = abs(buf[-1][0] - buf[0][0])
    return zero_crossings >= 2 and 0.001 < avg_disp < 0.01 and horiz_disp < 0.02


@app.post("/analyze")
async def analyze_video(stance: str = Form(...), video: UploadFile = File(...)):
    """
    NOTE:
    - Duplicate upload blocking is intentionally REMOVED (no 409 Conflict).
    - Mediapipe import is done inside this handler so the server can boot even if
      mediapipe isn't installed yet. If it's missing, you'll get a clear error.
    """
    if not video:
        raise HTTPException(status_code=400, detail="No video uploaded.")

    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video file.")

    # (Duplicate check removed on purpose)

    tmp_path = None
    try:
        # Lazy import so server can start without mediapipe installed
        try:
            import mediapipe as mp
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Mediapipe not available on server. Install it to analyze videos. ({e})",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video (codec/format).")

        # -------- Event detection timing (from FPS) --------
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Non-pumping: emit once per event; hold until different maneuver or neutral gap
        STABLE_SEC = 0.20       # require ~0.20s stable label
        NEUTRAL_RESET_SEC = 0.25  # need ~0.25s neutral to clear active move

        # Pumping: count one per full oscillation (up+down) with min interval + amplitude
        PUMP_MIN_INTERVAL_SEC = 0.35
        PUMP_MIN_Y_RANGE = 0.010

        MIN_STABLE_FRAMES = max(4, int(fps * STABLE_SEC))
        NEUTRAL_RESET_FRAMES = max(1, int(fps * NEUTRAL_RESET_SEC))
        PUMP_MIN_INTERVAL_FR = max(1, int(fps * PUMP_MIN_INTERVAL_SEC))

        mp_pose = mp.solutions.pose
        log = []
        buffer = []
        is_frontside = (stance == "f")

        # -------- State --------
        frame_idx = 0

        # Non-pumping
        active_move = ""       # current non-pumping maneuver
        stable_label = ""      # candidate stabilizing
        stable_count = 0
        last_neutral_frame = -10**9

        # Pumping (two sign changes = one pump)
        last_vy_sign = 0
        pump_phase = 0
        last_pump_emit_frame = -10**9

        def classify(buf, lms):
            if len(buf) < 10:
                return ""

            x0, y0 = buf[0]
            x5, y5 = buf[5]
            x9, y9 = buf[-1]
            dx_total = x9 - x0
            dy_total = y9 - y0
            dx_mid = x5 - x0
            vy = buf[-1][1] - buf[-2][1]

            lf = get_point(lms, mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
            rf = get_point(lms, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
            front = lf if is_frontside else rf
            back = rf if is_frontside else lf
            foot_sep = abs(front[0] - back[0])

            if detect_pumping(buf):
                return "Pumping"
            if dy_total < -0.035 and foot_sep > 0.035 and vy < -0.005:
                return "Bottom Turn"
            if dy_total > 0.035 and foot_sep > 0.035 and vy > 0.005:
                return "Top Turn"
            if (
                abs(dx_total) > 0.04
                and np.sign(dx_total) != np.sign(dx_mid)
                and foot_sep > 0.035
                and abs(dy_total) < 0.025
            ):
                return "Cutback"
            return ""

        # -------- Main loop --------
        with mp_pose.Pose(
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if not results.pose_landmarks:
                    # treat as neutral
                    last_neutral_frame = frame_idx
                    frame_idx += 1
                    continue

                lms = results.pose_landmarks.landmark
                center = get_point(lms, mp_pose.PoseLandmark.LEFT_HIP)
                buffer.append(center)
                if len(buffer) > 12:
                    buffer.pop(0)

                label = classify(buffer, lms)

                # ----- Neutral bookkeeping -----
                if label == "":
                    last_neutral_frame = frame_idx

                # ----- Pumping: count per full oscillation -----
                if len(buffer) >= 2:
                    vy = buffer[-1][1] - buffer[-2][1]
                    vy_sign = 1 if vy > 0 else (-1 if vy < 0 else 0)

                    y_vals = [p[1] for p in buffer]
                    y_range = (max(y_vals) - min(y_vals)) if y_vals else 0.0

                    if label == "Pumping":
                        if vy_sign != 0 and vy_sign != last_vy_sign:
                            pump_phase += 1  # sign flip
                            last_vy_sign = vy_sign
                        if (
                            pump_phase >= 2
                            and (frame_idx - last_pump_emit_frame) >= PUMP_MIN_INTERVAL_FR
                            and y_range >= PUMP_MIN_Y_RANGE
                        ):
                            log.append("Pumping")
                            last_pump_emit_frame = frame_idx
                            pump_phase = 0
                    else:
                        if (frame_idx - last_neutral_frame) >= NEUTRAL_RESET_FRAMES:
                            pump_phase = 0

                    if vy_sign != 0:
                        last_vy_sign = vy_sign

                # ----- Non-pumping: emit once, hold until different or neutral gap -----
                if label != "" and label != "Pumping":
                    if label == stable_label:
                        stable_count += 1
                    else:
                        stable_label = label
                        stable_count = 1

                    # clear active move after neutral gap
                    if (frame_idx - last_neutral_frame) >= NEUTRAL_RESET_FRAMES:
                        active_move = ""

                    if stable_count >= MIN_STABLE_FRAMES:
                        if active_move == "":
                            log.append(label)
                            active_move = label
                        elif active_move != label:
                            log.append(label)
                            active_move = label
                elif label == "":
                    # reset stabilization in neutral
                    stable_label = ""
                    stable_count = 0

                frame_idx += 1

        cap.release()

        score = wave_score(log)
        tips = "Try to maintain balance through transitions and keep knees bent for control."
        return {"score": score, "maneuvers": log, "tips": tips}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

