import cv2
import math
import time
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime
class VigilusEngine:
    def __init__(self, model_path: str):
        # --------------------------------
        # MediaPipe Face Landmarker
        # --------------------------------
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=2
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        # --------------------------------
        # Monitoring Mode
        # --------------------------------
        self.monitoring_mode = "Basic"
        self.set_mode(self.monitoring_mode)

    def set_mode(self, mode):
        self.monitoring_mode = mode
        if mode == "Basic":
            self.gaze_h_sensitivity = 3
            self.pitch_sensitivity = 3
            self.emotion_stability = 1.0
            self.gaze_stability = 0.8
            # Emotion Thresholds (Leniency)
            self.drowsy_eye_limit = 4.2      # Lower = harder to trigger drowsiness
            self.drowsy_mouth_limit = 4.0
            self.drowsy_lip_v_limit = 10.0
            self.surprised_eye_limit = 6.0   # Higher = harder to trigger surprise
            self.surprised_mouth_limit = 8.5
            self.confused_eye_limit = 7.0
            self.happy_lip_width_limit = 40.0
        elif mode == "Intermediate":
            self.gaze_h_sensitivity = 5
            self.pitch_sensitivity = 5
            self.emotion_stability = 0.5
            self.gaze_stability = 0.4
            # Emotion Thresholds (Standard)
            self.drowsy_eye_limit = 4.8
            self.drowsy_mouth_limit = 5.0
            self.drowsy_lip_v_limit = 12.0
            self.surprised_eye_limit = 5.5
            self.surprised_mouth_limit = 7.0
            self.confused_eye_limit = 6.5
            self.happy_lip_width_limit = 37.0
        elif mode == "Strict":
            self.gaze_h_sensitivity = 8
            self.pitch_sensitivity = 7
            self.emotion_stability = 0.2
            self.gaze_stability = 0.2
            # Emotion Thresholds (Sensitive)
            self.drowsy_eye_limit = 5.2      # Higher = easier to trigger drowsiness
            self.drowsy_mouth_limit = 6.0
            self.drowsy_lip_v_limit = 14.0
            self.surprised_eye_limit = 5.2
            self.surprised_mouth_limit = 6.5
            self.confused_eye_limit = 6.0
            self.happy_lip_width_limit = 35.0
        # --------------------------------
        # Runtime State
        # --------------------------------
        self.last_emotion = None
        self.emotion_start = None
        self.stable_emotion = "Emotion: Neutral"
        self.last_gaze = None
        self.gaze_start = None
        self.stable_gaze = "Center"
        self.attention_state = "ATTENTIVE"
        self.attention_start = time.time()
        self.cognitive_state = "NORMAL"
        self.cognitive_start = time.time()
        
        # Logging State
        self.last_logged_event = None
        self.last_log_time = 0
    # --------------------------------
    # Update Config from UI
    # --------------------------------
    def update_config(self, **kwargs):
        """
        Called directly from PySide6 UI.

        Example:
            engine.update_config(
                gaze_h_sensitivity=6,
                pitch_sensitivity=4,
                emotion_stability=0.8,
                gaze_stability=0.5,
                monitoring_mode="Strict"
            )
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # --------------------------------
    # Utilities
    # --------------------------------
    @staticmethod
    def dist(a, b):
        return math.dist(a, b)

    def log_distraction(self, event_type):
        """
        Logs a distraction event to distractions.log with a timestamp.
        Prevents spamming by logging same event type only every 3 seconds.
        """
        now = time.time()
        if event_type == self.last_logged_event and (now - self.last_log_time < 3):
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] MODE: {self.monitoring_mode} | EVENT: {event_type}\n"
        
        try:
            with open("distractions.log", "a") as f:
                f.write(log_entry)
            self.last_logged_event = event_type
            self.last_log_time = now
            print(f"LOGGED: {event_type}")
        except Exception as e:
            print(f"Error writing to log: {e}")

    # --------------------------------
    # Threshold Mapping
    # --------------------------------
    def _map_gaze_threshold(self):
        """
        Horizontal gaze sensitivity:
        Slider 1–10 → threshold
        Higher value = more sensitive
        """
        base = 0.006
        return base * (11 - self.gaze_h_sensitivity) / 10.0

    def _map_pitch_thresholds(self):
        """
        Vertical gaze via head pitch.
        Higher sensitivity = tighter up/down detection.
        """
        s = self.pitch_sensitivity
        up = 0.45 - (s * 0.01)
        down = 0.55 + (s * 0.01)
        return up, down

    # --------------------------------
    # MAIN PROCESS FUNCTION
    # --------------------------------
    def process(self, frame):
        """
        Input : BGR OpenCV frame
        Output: annotated_frame, data_dict
        """

        now = time.time()
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_present = False
        current_emotion = "Emotion: Neutral"
        final_gaze = "Center"

        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self.face_landmarker.detect(mp_image)

        # Dynamic thresholds
        gaze_thresh = self._map_gaze_threshold()
        pitch_up, pitch_down = self._map_pitch_thresholds()

        if result.face_landmarks:
            print(f"DEBUG: Detected {len(result.face_landmarks)} faces")
            face_present = True
            lm = result.face_landmarks[0]

            def pt(i):
                return (lm[i].x * w, lm[i].y * h)

            # ---------- Head Pitch (Vertical Gaze)
            forehead, nose, chin = pt(10), pt(1), pt(152)
            face_height = self.dist(forehead, chin) + 1e-6
            pitch_ratio = (nose[1] - forehead[1]) / face_height

            gaze_v = "Center"
            if pitch_ratio < pitch_up:
                gaze_v = "Up"
            elif pitch_ratio > pitch_down:
                gaze_v = "Down"

            # ---------- Horizontal Gaze
            left_eye = self.dist(pt(159), pt(145))
            right_eye = self.dist(pt(386), pt(374))
            diff = (left_eye - right_eye) / face_height

            gaze_h = "Center"
            if diff > gaze_thresh:
                gaze_h = "Right"
            elif diff < -gaze_thresh:
                gaze_h = "Left"

            # ---------- Final 9-direction gaze
            if gaze_h == "Center" and gaze_v == "Center":
                final_gaze = "Center"
            elif gaze_h != "Center" and gaze_v == "Center":
                final_gaze = gaze_h
            elif gaze_h == "Center" and gaze_v != "Center":
                final_gaze = gaze_v
            else:
                final_gaze = f"{gaze_v}-{gaze_h}"

            # ---------- Emotion Logic
            mouth_open = self.dist(pt(13), pt(14))
            eye_avg = (left_eye + right_eye) / 2

            eye_score = (eye_avg / face_height) * 100
            mouth_score = (mouth_open / face_height) * 100

            lip_left, lip_right = pt(61), pt(291)
            mouth_width = self.dist(lip_left, lip_right)

            face_left, face_right = pt(234), pt(454)
            face_width = self.dist(face_left, face_right)

            lip_width_score = (mouth_width / face_width) * 100
            lip_vertical_score = (mouth_open / mouth_width) * 100

            if eye_score < self.drowsy_eye_limit and mouth_score < self.drowsy_mouth_limit and lip_vertical_score < self.drowsy_lip_v_limit:
                current_emotion = "Emotion: Drowsy"
            elif eye_score > self.surprised_eye_limit and mouth_score > self.surprised_mouth_limit and lip_vertical_score > 20:
                current_emotion = "Emotion: Surprised"
            elif eye_score > self.confused_eye_limit and mouth_score < 5:
                current_emotion = "Emotion: Confused"
            elif eye_score > 4.7 and lip_width_score > self.happy_lip_width_limit:
                current_emotion = "Emotion: Happy"

        # ---------- Emotion Stability
        if current_emotion != self.last_emotion:
            self.last_emotion = current_emotion
            self.emotion_start = now
        elif self.emotion_start and now - self.emotion_start >= self.emotion_stability:
            self.stable_emotion = current_emotion

        # ---------- Gaze Stability
        if final_gaze != self.last_gaze:
            self.last_gaze = final_gaze
            self.gaze_start = now
        elif self.gaze_start and now - self.gaze_start >= self.gaze_stability:
            self.stable_gaze = final_gaze

        # ---------- Attention
        attention = "ATTENTIVE" if self.stable_gaze == "Center" else "DISTRACTED"

        # ---------- Cognitive
        if self.stable_emotion == "Emotion: Drowsy":
            cognitive = "FATIGUED"
        elif attention != "ATTENTIVE":
            cognitive = "DISTRACTED"
        else:
            cognitive = "FOCUSED"

        # ---------- Data for UI
        face_count = len(result.face_landmarks) if result.face_landmarks else 0
        data = {
            "face_present": face_present,
            "face_count": face_count,
            "emotion": self.stable_emotion,
            "gaze": self.stable_gaze,
            "attention": attention,
            "cognitive": cognitive,
            "mode": self.monitoring_mode
        }

        # ---------- Log Distractions
        if attention == "DISTRACTED":
            self.log_distraction("GAZE DISTRACTED")
        elif cognitive == "FATIGUED":
            self.log_distraction("DROWSY/FATIGUED")
        elif not face_present:
            self.log_distraction("NO FACE DETECTED")
        elif face_count > 1:
            self.log_distraction("MULTIPLE FACES DETECTED")

        return frame, data
