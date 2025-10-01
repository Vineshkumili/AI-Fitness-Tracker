import time
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np

# ---------- Utility functions ----------

def calc_angle(a, b, c):
    """Return angle (degrees) at point b formed by points a-b-c.
    Points should be (x, y) pairs in image coordinates (or normalized coords).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Prevent division by zero
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return None

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def get_landmark_coords(landmarks, idx, image_w, image_h):
    lm = landmarks[idx]
    return (int(lm.x * image_w), int(lm.y * image_h))


# ---------- Exercise logic implementations ----------

class ExerciseState:
    def __init__(self):
        self.count = 0
        self.stage = None  # 'up'/'down' or similar
        self.last_time = 0
        self.timer_running = False
        self.timer_start = None


# Each function accepts (landmarks, image_w, image_h, state) and returns updated state


def detect_pushup(landmarks, w, h, state: ExerciseState):
    # Use right elbow angle (shoulder-elbow-wrist). You can optionally average left+right.
    try:
        r_sh = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
        r_el = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value, w, h)
        r_wr = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value, w, h)
    except Exception:
        return state

    angle = calc_angle(r_sh, r_el, r_wr)
    if angle is None:
        return state

    # thresholds - tune as needed
    down_thresh = 90
    up_thresh = 160

    if angle < down_thresh:
        state.stage = 'down'
    if angle > up_thresh and state.stage == 'down':
        state.count += 1
        state.stage = 'up'
    return state


def detect_situp(landmarks, w, h, state: ExerciseState):
    # Use hip angle (shoulder-hip-knee) on right side
    try:
        r_sh = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
        r_hip = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, w, h)
        r_knee = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, w, h)
    except Exception:
        return state

    angle = calc_angle(r_sh, r_hip, r_knee)
    if angle is None:
        return state

    # Sit-up: when hip angle decreases (curl) then extends
    curl_thresh = 90
    extend_thresh = 160

    if angle < curl_thresh:
        state.stage = 'curl'
    if angle > extend_thresh and state.stage == 'curl':
        state.count += 1
        state.stage = 'extended'
    return state


def detect_squat(landmarks, w, h, state: ExerciseState):
    # Use knee angle (hip-knee-ankle) on right side
    try:
        r_hip = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, w, h)
        r_knee = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, w, h)
        r_ank = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value, w, h)
    except Exception:
        return state

    angle = calc_angle(r_hip, r_knee, r_ank)
    if angle is None:
        return state

    down_thresh = 100  # knee angle less than this considered squat down
    up_thresh = 160

    if angle < down_thresh:
        state.stage = 'down'
    if angle > up_thresh and state.stage == 'down':
        state.count += 1
        state.stage = 'up'
    return state


def detect_jumping_jack(landmarks, w, h, state: ExerciseState):
    # Simple heuristic: arms above head AND legs apart
    try:
        l_wr = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.LEFT_WRIST.value, w, h)
        r_wr = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value, w, h)
        l_sh = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
        r_sh = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
        l_ank = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value, w, h)
        r_ank = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value, w, h)
    except Exception:
        return state

    # arms above head if both wrists y < shoulders y (remember y increases downward)
    arms_up = (l_wr[1] < l_sh[1] - 10) and (r_wr[1] < r_sh[1] - 10)

    # legs apart: horizontal distance between ankles greater than some fraction of image width
    legs_dist = abs(l_ank[0] - r_ank[0])
    legs_apart = legs_dist > (w * 0.25)

    if arms_up and legs_apart:
        if state.stage != 'open':
            state.stage = 'open'
    else:
        if state.stage == 'open':
            state.count += 1
            state.stage = 'closed'
    return state


def detect_lunge(landmarks, w, h, state: ExerciseState):
    # Heuristic: detect significant forward knee bend on one leg (right leg used here)
    try:
        r_hip = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, w, h)
        r_knee = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, w, h)
        r_ank = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value, w, h)
    except Exception:
        return state

    angle = calc_angle(r_hip, r_knee, r_ank)
    if angle is None:
        return state

    down_thresh = 100
    up_thresh = 160

    if angle < down_thresh:
        state.stage = 'down'
    if angle > up_thresh and state.stage == 'down':
        state.count += 1
        state.stage = 'up'
    return state


def detect_plank(landmarks, w, h, state: ExerciseState):
    # Check body straightness via hip angle (shoulder-hip-ankle on right side)
    try:
        r_sh = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
        r_hip = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, w, h)
        r_ank = get_landmark_coords(landmarks, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value, w, h)
    except Exception:
        return state

    angle = calc_angle(r_sh, r_hip, r_ank)
    if angle is None:
        return state

    straight_min = 160
    straight_max = 200  # safe upper clamp

    is_straight = straight_min <= angle <= straight_max

    if is_straight:
        if not state.timer_running:
            state.timer_running = True
            state.timer_start = time.time()
    else:
        if state.timer_running:
            # stop and record elapsed in last_time
            state.timer_running = False
            state.last_time = time.time() - state.timer_start
            state.timer_start = None
    return state


# ---------- Main loop ----------

EXERCISE_MAP = {
    '1': ('Push-ups', detect_pushup),
    '2': ('Sit-ups', detect_situp),
    '3': ('Squats', detect_squat),
    '4': ('Jumping Jacks', detect_jumping_jack),
    '5': ('Lunges', detect_lunge),
    '6': ('Plank', detect_plank),
}


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open webcam.')
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    selected = '1'  # default to push-ups
    states = defaultdict(ExerciseState)

    print('Controls: 1-Pushups 2-Situps 3-Squats 4-JumpingJacks 5-Lunges 6-Plank r-reset q-quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_h, image_w = frame.shape[:2]
        # flip mirror view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            exercise_name, func = EXERCISE_MAP[selected]
            state = states[selected]
            state_before = state.count

            # Call detector
            state = func(lm, image_w, image_h, state)
            states[selected] = state

            # Overlay info
            cv2.putText(frame, f'Exercise: {exercise_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if selected != '6':
                cv2.putText(frame, f'Reps: {state.count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            else:
                # Plank - show timer
                if state.timer_running:
                    elapsed = time.time() - state.timer_start
                    cv2.putText(frame, f'Plank: {elapsed:.1f}s', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                else:
                    last = state.last_time
                    cv2.putText(frame, f'Plank last: {last:.1f}s', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # Provide small feedback on motion
            # show angle for debug depending on exercise
            if selected == '1':
                # elbow angle
                try:
                    r_sh = get_landmark_coords(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, image_w, image_h)
                    r_el = get_landmark_coords(lm, mp_pose.PoseLandmark.RIGHT_ELBOW.value, image_w, image_h)
                    r_wr = get_landmark_coords(lm, mp_pose.PoseLandmark.RIGHT_WRIST.value, image_w, image_h)
                    angle = calc_angle(r_sh, r_el, r_wr)
                    if angle is not None:
                        cv2.putText(frame, f'Elbow angle: {int(angle)}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                except Exception:
                    pass

        else:
            cv2.putText(frame, 'No person detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, f'Selected [{selected}]: {EXERCISE_MAP[selected][0]}', (10, image_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('AI Fitness Tracker', frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            c = chr(key)
            if c in EXERCISE_MAP.keys():
                selected = c
                print('Selected:', EXERCISE_MAP[selected][0])
            elif c == 'r':
                states = defaultdict(ExerciseState)
                print('Reset counters/timers')
            elif c in ('q', '\x1b'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
