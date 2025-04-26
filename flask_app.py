# flask-app.py
import os
import cv2
import base64
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
import logging
import time

# --- Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_key_12345")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize MediaPipe solutions ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# --- Initialize MediaPipe Models ---
hands = None
try:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    logger.info("MediaPipe models initialized (Hands max_num=2).")
except Exception as e:
    logger.error(f"Error initializing MediaPipe: {e}", exc_info=True)

# --- Global Variables ---
active_gestures = {}
prev_hand1_pos_norm = None
prev_hand1_timestamp = None
was_fist1_closed_prev = False
was_fist1_pinched_prev = False
prev_two_hand_distance = None
last_two_hand_time = None
ZOOM_THRESHOLD_INCREASE = 1.05

# --- Helper: Finger Extended Check ---
def is_finger_extended(tip, base, wrist, threshold_mult=1.1):
    try:
        if not all(key in tip for key in ['px', 'py']) or \
           not all(key in base for key in ['px', 'py']) or \
           not all(key in wrist for key in ['px', 'py']):
            return False

        dist_tip_wrist_px = np.linalg.norm(np.array((tip['px'], tip['py'])) - np.array((wrist['px'], wrist['py'])))
        dist_base_wrist_px = np.linalg.norm(np.array((base['px'], base['py'])) - np.array((wrist['px'], wrist['py'])))

        return False if dist_base_wrist_px < 1e-6 else dist_tip_wrist_px > dist_base_wrist_px * threshold_mult
    except Exception as e:
        logger.warning(f"Error in is_finger_extended: {e}")
        return False

def detect_hand_gestures(frame):
    global was_fist1_closed_prev, prev_hand1_pos_norm, prev_hand1_timestamp, was_fist1_pinched_prev
    global prev_two_hand_distance, last_two_hand_time

    if frame is None or hands is None:
        was_fist1_pinched_prev = False
        was_fist1_closed_prev = False
        prev_hand1_pos_norm = None
        prev_hand1_timestamp = None
        prev_two_hand_distance = None
        last_two_hand_time = None
        return frame, {}, None, False, None, 0

    frame_height, frame_width = frame.shape[:2]
    frame.flags.writeable = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    frame.flags.writeable = True

    gestures = {"click": False, "rotate": False, "zoom_in": False}
    rotating_hand_position_norm = None
    flick_detected_this_frame = False
    flick_velocity = None
    hand_count = 0
    all_hands_data = []
    current_time = time.time()

    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_drawing_styles.get_default_hand_landmarks_style(),
                                   mp_drawing_styles.get_default_hand_connections_style())

            landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z,
                         'px': int(lm.x * frame_width), 'py': int(lm.y * frame_height)}
                        for lm in hand_landmarks.landmark]

            if len(landmarks) < 21:
                continue

            try:
                thumb_tip=landmarks[4]; index_tip=landmarks[8]; middle_tip=landmarks[12]
                ring_tip=landmarks[16]; pinky_tip=landmarks[20]
                thumb_base=landmarks[1]; index_base=landmarks[5]; middle_base=landmarks[9]
                ring_base=landmarks[13]; pinky_base=landmarks[17]; wrist=landmarks[0]

                thumb_extended=is_finger_extended(thumb_tip,thumb_base,wrist)
                index_extended=is_finger_extended(index_tip,index_base,wrist)
                middle_extended=is_finger_extended(middle_tip,middle_base,wrist)
                ring_extended=is_finger_extended(ring_tip,ring_base,wrist)
                pinky_extended=is_finger_extended(pinky_tip,pinky_base,wrist)

                is_splayed = index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended
                is_fist_closed_now = not index_extended and not middle_extended and not ring_extended and not pinky_extended
                is_pinch = thumb_extended and index_extended

                current_hand_pos_norm = {'x': wrist['x'], 'y': wrist['y']}
                hand_data = {'wrist': wrist, 'is_splayed': is_splayed, 'is_fist': is_fist_closed_now, 'is_pinch': is_pinch}
                all_hands_data.append(hand_data)

                thumb_tip_px=(thumb_tip['px'], thumb_tip['py'])
                index_tip_px=(index_tip['px'], index_tip['py'])
                thumb_index_dist=np.linalg.norm(np.array(thumb_tip_px) - np.array(index_tip_px))
                hand_size_metric=np.linalg.norm(np.array((wrist['px'],wrist['py'])) - np.array((middle_base['px'],middle_base['py'])))
                pinch_threshold = (hand_size_metric * 0.35) if hand_size_metric > 1e-6 else 11.5

                if is_pinch and thumb_index_dist < pinch_threshold:
                    gestures["rotate"] = True
                    rotating_hand_position_norm = current_hand_pos_norm

                if is_fist_closed_now and rotating_hand_position_norm is None:
                    gestures["rotate"] = True
                    rotating_hand_position_norm = current_hand_pos_norm

                if hand_idx == 0:
                    if was_fist1_closed_prev and not is_fist_closed_now:
                        flick_detected_this_frame = True
                        if prev_hand1_pos_norm and prev_hand1_timestamp:
                            delta_time = current_time - prev_hand1_timestamp
                            if delta_time > 0.01:
                                delta_x = current_hand_pos_norm['x'] - prev_hand1_pos_norm['x']
                                delta_y = current_hand_pos_norm['y'] - prev_hand1_pos_norm['y']
                                flick_velocity = {'x': delta_x / delta_time, 'y': delta_y / delta_time}
                                logger.info(f"Flick Hand 1! V: {flick_velocity}")
                    was_fist1_closed_prev = is_fist_closed_now
                    prev_hand1_pos_norm = current_hand_pos_norm
                    prev_hand1_timestamp = current_time

            except Exception as e:
                logger.warning(f"Error processing landmarks hand {hand_idx}: {e}")

        if hand_count == 2 and len(all_hands_data) == 2:
            try:
                hand1 = all_hands_data[0]
                hand2 = all_hands_data[1]
                if hand1['is_splayed'] and hand2['is_splayed']:
                    wrist1_norm=(hand1['wrist']['x'], hand1['wrist']['y'])
                    wrist2_norm=(hand2['wrist']['x'], hand2['wrist']['y'])
                    current_distance = np.linalg.norm(np.array(wrist1_norm) - np.array(wrist2_norm))
                    if prev_two_hand_distance is not None and last_two_hand_time is not None:
                        time_diff = current_time - last_two_hand_time
                        if 0.02 < time_diff < 0.5 and current_distance > prev_two_hand_distance * ZOOM_THRESHOLD_INCREASE:
                            gestures["zoom_in"] = True
                    prev_two_hand_distance = current_distance
                    last_two_hand_time = current_time
                else:
                    prev_two_hand_distance = None
                    last_two_hand_time = None
            except Exception as e:
                logger.warning(f"Error in 2-hand logic: {e}")
                prev_two_hand_distance = None
                last_two_hand_time = None
    else:
        was_fist1_closed_prev = False
        prev_hand1_pos_norm = None
        prev_hand1_timestamp = None
        was_fist1_pinched_prev = False
        prev_two_hand_distance = None
        last_two_hand_time = None

    return frame, gestures, rotating_hand_position_norm, flick_detected_this_frame, flick_velocity, hand_count

# --- Utility Functions ---
def process_image_from_data_url(data_url):
    try:
        if 'base64,' in data_url:
            data_url = data_url.split('base64,')[1]
        image_data = base64.b64decode(data_url)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("imdecode returned None.")
        return img
    except Exception as e:
        logger.error(f"Decode Error: {e}", exc_info=True)
        return None

def convert_image_to_data_url(image):
    if image is None:
        logger.warning("convert None image.")
        return None
    try:
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            logger.error("imencode failed.")
            return None
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Encode Error: {e}", exc_info=True)
        return None

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    global active_gestures
    data = request.json
    img_data = data.get('image') if data else None
    if not img_data:
        return jsonify({'error': 'No image data'}), 400

    frame = process_image_from_data_url(img_data)
    if frame is None:
        return jsonify({'error': 'Decode failed'}), 400

    frame = cv2.flip(frame, 1)
    hand_frame, gestures, rot_hand_pos_norm, flick_detected, flick_velocity, hand_count = detect_hand_gestures(frame.copy())

    result_frame = hand_frame
    active_gestures = {k: v for k, v in gestures.items() if v}
    processed_img_data = convert_image_to_data_url(result_frame)
    if processed_img_data is None:
        return jsonify({'error': 'Encode failed'}), 500

    response_data = {
        'processed_image': processed_img_data,
        'gestures': active_gestures,
        'rotating_hand_position': rot_hand_pos_norm,
        'flick_detected': flick_detected,
        'flick_velocity': flick_velocity,
        'hand_count': hand_count
    }
    return jsonify(response_data)

@app.route('/calibrate', methods=['POST'])
def calibrate_endpoint():
    global calibrated, calibration_data, calibration_matrix, prev_x_percent, prev_y_percent
    data = request.json
    action = data.get('action') if data else None
    if not action:
        logger.error("Calibration request missing action.")
        return jsonify({'status': 'error', 'message': 'No action specified'}), 400

    if action == 'start':
        calibrated = False
        calibration_data = []
        calibration_matrix = None
        prev_x_percent = 50.0
        prev_y_percent = 50.0
        logger.info("Calibration started.")
        return jsonify({'status': 'started'})

    elif action == 'capture':
        try:
            sp = data.get('screen_point')
            ep = data.get('eye_position')
            assert isinstance(sp, dict) and 'x' in sp and 'y' in sp
            assert isinstance(ep, dict) and 'x' in ep and 'y' in ep
            assert isinstance(sp['x'], (int, float))

            calibration_data.append({'screen_point': (sp['x'], sp['y']), 'eye_position': (ep['x'], ep['y'])})
            logger.info(f"Captured calib point {len(calibration_data)}")
            return jsonify({'status': 'point_captured', 'points_captured': len(calibration_data)})
        except (AssertionError, TypeError, KeyError) as e:
            logger.error(f"Invalid capture data format: {data}. Error: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': 'Invalid capture data format'}), 400
        except Exception as e:
            logger.error(f"Unexpected error during capture: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': 'Unexpected error during capture processing'}), 500

    elif action == 'complete':
        min_points = 4
        if len(calibration_data) < min_points:
            logger.error(f"Calibration failed: Need {min_points} points, got {len(calibration_data)}.")
            return jsonify({'status': 'error', 'message': f'Need at least {min_points} calibration points'}), 400
        try:
            src = np.float32([d['eye_position'] for d in calibration_data]).reshape(-1, 1, 2)
            dst = np.float32([d['screen_point'] for d in calibration_data]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if matrix is None:
                 raise ValueError("findHomography returned None (points might be collinear/degenerate)")

            calibration_matrix = matrix
            calibrated = True
            logger.info(f"Calibration successful ({len(calibration_data)} pts).")
            return jsonify({'status': 'success', 'message': f'Calibration complete ({len(calibration_data)} pts).'})
        except Exception as e:
            logger.error(f"Calibration calculation failed: {e}", exc_info=True)
            calibrated = False
            calibration_matrix = None
            return jsonify({'status': 'error', 'message': f'Calibration calculation failed: {e}'}), 500
    else:
        logger.error(f"Invalid action '{action}' received in /calibrate")
        return jsonify({'status': 'error', 'message': 'Invalid action specified'}), 400

if __name__ == '__main__':
    print("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
  pass