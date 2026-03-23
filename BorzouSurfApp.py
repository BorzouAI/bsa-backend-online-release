# =========================
# BorzouSurfApp.py
# Backend: Surf Analyzer
# =========================

import os
import hashlib
import sys
import tempfile

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np

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


# Hit: https://bsa-backend-online-release-cmk9.onrender.com/debug/mediapipe
@app.get("/debug/mediapipe")
def debug_mediapipe():
    try:
        import mediapipe as mp
        return {
            "python": sys.version,
            "mediapipe_version": getattr(mp, "__version__", "unknown"),
            "mediapipe_file": getattr(mp, "__file__", "unknown"),
            "has_solutions": hasattr(mp, "solutions"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import mediapipe: {e}")


# --- Scoring config ---
# Conservative v1 scoring:
# - Pumps should help a little, not dominate.
# - Turns/cutbacks matter more.
# - Each next maneuver is worth less.
# - Repeating the same maneuver back-to-back is penalized.
# - 9+ should be very hard to get.
BASE_POINTS = {
    "Bottom Turn": 1.6,
    "Top Turn": 2.0,
    "Cutback": 2.6,
    "Pumping": 0.30,
}

# Every next logged maneuver gets reduced.
SEQUENCE_DECAY = 0.72

# Later maneuvers still count a bit, but not much.
MIN_SEQUENCE_MULTIPLIER = 0.18

# Back-to-back same move penalty.
REPEAT_MOVE_MULTIPLIER = 0.55

# Slight bonus for mixing up non-pumping maneuvers.
VARIETY_BONUS = 0.20

# Soft-cap region: once raw score crosses this, gains compress heavily.
SOFT_CAP_START = 8.0
SOFT_CAP_FACTOR = 0.28


def wave_score(maneuver_log):
    if not maneuver_log:
        return 0.0

    total_score = 0.0
    seen_major_moves = set()
    prev_move = None

    for i, maneuver in enumerate(maneuver_log):
        base = BASE_POINTS.get(maneuver, 0.0)
        if base <= 0:
            continue

        seq_multiplier = max(SEQUENCE_DECAY ** i, MIN_SEQUENCE_MULTIPLIER)
        move_score = base * seq_multiplier

        # Penalize repeating the exact same move back-to-back.
        if prev_move == maneuver:
            move_score *= REPEAT_MOVE_MULTIPLIER

        # Small reward for adding a new major maneuver type to the wave.
        if maneuver != "Pumping" and maneuver not in seen_major_moves:
            move_score += VARIETY_BONUS
            seen_major_moves.add(maneuver)

        total_score += move_score
        prev_move = maneuver

    # Soft cap near the top so 9+ is very hard.
    if total_score > SOFT_CAP_START:
        total_score = SOFT_CAP_START + (total_score - SOFT_CAP_START) * SOFT_CAP_FACTOR

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


def _get_mp_pose_module():
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "Mediapipe imported but mp.solutions is missing. "
            f"mp.__version__={getattr(mp, '__version__', 'unknown')} "
            f"mp.__file__={getattr(mp, '__file__', 'unknown')}"
        )
    return mp.solutions.pose


@app.post("/analyze")
def analyze_video(stance: str = Form(...), video: UploadFile = File(...)):
    if not video:
        raise HTTPException(status_code=400, detail="No video uploaded.")

    try:
        video_bytes = video.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video file.")

    tmp_path = None
    try:
        try:
            mp_pose = _get_mp_pose_module()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Server error: {e}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video (codec/format).")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # -------- Speed controls --------
        TARGET_ANALYSIS_FPS = 5.0
        FRAME_STEP = max(1, int(round(fps / TARGET_ANALYSIS_FPS)))
        MAX_ANALYZED_FRAMES = 900

        # -------- Event timing --------
        STABLE_SEC = 0.20
        NEUTRAL_RESET_SEC = 0.25
        PUMP_MIN_INTERVAL_SEC = 0.35
        PUMP_MIN_Y_RANGE = 0.010

        analysis_fps = fps / FRAME_STEP
        MIN_STABLE_FRAMES = max(2, int(analysis_fps * STABLE_SEC))
        NEUTRAL_RESET_FRAMES = max(1, int(analysis_fps * NEUTRAL_RESET_SEC))
        PUMP_MIN_INTERVAL_FR = max(1, int(analysis_fps * PUMP_MIN_INTERVAL_SEC))

        log = []
        buffer = []
        is_frontside = (stance == "f")

        frame_idx = 0
        analyzed_frames = 0

        active_move = ""
        stable_label = ""
        stable_count = 0
        last_neutral_frame = -10**9

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

        with mp_pose.Pose(
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % FRAME_STEP != 0:
                    frame_idx += 1
                    continue

                analyzed_frames += 1
                if analyzed_frames > MAX_ANALYZED_FRAMES:
                    break

                h, w = frame.shape[:2]
                if w > 640:
                    new_w = 640
                    new_h = int(h * (new_w / w))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if not results.pose_landmarks:
                    last_neutral_frame = analyzed_frames
                    frame_idx += 1
                    continue

                lms = results.pose_landmarks.landmark
                center = get_point(lms, mp_pose.PoseLandmark.LEFT_HIP)
                buffer.append(center)
                if len(buffer) > 12:
                    buffer.pop(0)

                label = classify(buffer, lms)

                if label == "":
                    last_neutral_frame = analyzed_frames

                # Pumping
                if len(buffer) >= 2:
                    vy = buffer[-1][1] - buffer[-2][1]
                    vy_sign = 1 if vy > 0 else (-1 if vy < 0 else 0)

                    y_vals = [p[1] for p in buffer]
                    y_range = (max(y_vals) - min(y_vals)) if y_vals else 0.0

                    if label == "Pumping":
                        if vy_sign != 0 and vy_sign != last_vy_sign:
                            pump_phase += 1
                            last_vy_sign = vy_sign
                        if (
                            pump_phase >= 2
                            and (analyzed_frames - last_pump_emit_frame) >= PUMP_MIN_INTERVAL_FR
                            and y_range >= PUMP_MIN_Y_RANGE
                        ):
                            log.append("Pumping")
                            last_pump_emit_frame = analyzed_frames
                            pump_phase = 0
                    else:
                        if (analyzed_frames - last_neutral_frame) >= NEUTRAL_RESET_FRAMES:
                            pump_phase = 0

                    if vy_sign != 0:
                        last_vy_sign = vy_sign

                # Non-pumping
                if label != "" and label != "Pumping":
                    if label == stable_label:
                        stable_count += 1
                    else:
                        stable_label = label
                        stable_count = 1

                    if (analyzed_frames - last_neutral_frame) >= NEUTRAL_RESET_FRAMES:
                        active_move = ""

                    if stable_count >= MIN_STABLE_FRAMES:
                        if active_move == "":
                            log.append(label)
                            active_move = label
                        elif active_move != label:
                            log.append(label)
                            active_move = label
                elif label == "":
                    stable_label = ""
                    stable_count = 0

                frame_idx += 1

        cap.release()

        score = wave_score(log)
        tips = "Try to maintain balance through transitions and keep knees bent for control."

        return {
            "score": score,
            "maneuvers": log,
            "tips": tips,
        }

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
