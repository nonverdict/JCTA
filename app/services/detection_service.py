import asyncio
import threading
import cv2
import numpy as np
import time
from ultralytics import YOLO
from pathlib import Path
from app.utils.logger import logger
import config


class CameraManager:
    def __init__(self, source, is_file=False):
        self._source = source
        self._is_file = is_file
        self._cv_cap = None
        self.source_description = "Video" if is_file else f"Webcam {source}"

    def start(self):
        self.release()
        self._cv_cap = cv2.VideoCapture(
            str(self._source) if self._is_file else self._source
        )
        return self._cv_cap.isOpened()

    def is_open(self):
        return self._cv_cap is not None and self._cv_cap.isOpened()

    def read(self):
        if not self.is_open():
            return None
        success, frame = self._cv_cap.read()
        return frame if success else None

    def release(self):
        if self._cv_cap is not None:
            self._cv_cap.release()
            self._cv_cap = None


class DetectionService:
    def __init__(self, app):
        self.app = app
        self.model = None
        self.tracked_objects = {}
        self.next_track_id = 0
        self.current_frame_index = 0
        self._lock = threading.Lock()
        self._lethargy_demo_target_id = None

    def next_frame_index(self):
        with self._lock:
            self.current_frame_index += 1
            return self.current_frame_index

    async def initialize(self):
        loop = asyncio.get_running_loop()
        self.model = await loop.run_in_executor(
            self.app["executor"], lambda: YOLO(config.YOLO_MODEL_PATH)
        )
        self.model.to("cpu")
        logger.info("YOLO Initialized on CPU.")

    def process_frame(self, frame_orig, frame_res, frame_idx):
        if self.model is None:
            return frame_res, []

        with self._lock:
            return self._process_frame_locked(frame_orig, frame_res, frame_idx)

    def _process_frame_locked(self, frame_orig, frame_res, frame_idx):
        results = self.model(
            frame_res,
            verbose=False,
            conf=config.YOLO_CONFIDENCE_THRESHOLD,
            iou=config.YOLO_IOU_THRESHOLD_NMS,
            classes=config.YOLO_TARGET_CLASSES,
        )

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    detections.append(
                        {
                            "bbox_center": (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2),
                            "bbox_xywh": (x1, y1, x2 - x1, y2 - y1),
                        }
                    )

        current_tracks = []
        temp_state = {}
        unmatched = list(range(len(detections)))

        for tid, data in self.tracked_objects.items():
            data["frames_unseen"] += 1
            best_idx, min_dist = -1, config.TRACKER_MAX_DISTANCE
            for i in unmatched:
                dist = np.linalg.norm(
                    np.array(data["bbox_center"])
                    - np.array(detections[i]["bbox_center"])
                )
                if dist < min_dist:
                    min_dist, best_idx = dist, i

            if best_idx != -1:
                det = detections[best_idx]
                data.update(
                    {
                        "bbox_center": det["bbox_center"],
                        "frames_unseen": 0,
                        "bbox_history": (data["bbox_history"] + [det["bbox_xywh"]])[
                            -10:
                        ],
                    }
                )
                current_tracks.append({"id": tid, "bbox_xywh": det["bbox_xywh"]})
                unmatched.remove(best_idx)
                temp_state[tid] = data
            elif data["frames_unseen"] <= config.MAX_UNSEEN_FRAMES_TRACKER:
                temp_state[tid] = data

        for i in unmatched:
            det = detections[i]
            temp_state[self.next_track_id] = {
                "bbox_center": det["bbox_center"],
                "creation_frame_index": frame_idx,
                "last_seen_frame": frame_idx,
                "frames_unseen": 0,
                "bbox_history": [det["bbox_xywh"]],
                "disease_history": [],
                "last_classified_frame": 0,
                "active_alert": None,
                "movement_history": [],
                "lethargy_start_time": 0,
                "classification_count": 0,
            }
            current_tracks.append(
                {"id": self.next_track_id, "bbox_xywh": det["bbox_xywh"]}
            )
            self.next_track_id = (self.next_track_id + 1) % 20000

        self.tracked_objects = temp_state

        # Update movement history for active tracks
        for track in current_tracks:
            tid = track["id"]
            data = self.tracked_objects[tid]
            if len(data["bbox_history"]) > 1:
                prev = np.array(data["bbox_history"][-2][:2]) + np.array(data["bbox_history"][-2][2:]) / 2
                curr = np.array(data["bbox_history"][-1][:2]) + np.array(data["bbox_history"][-1][2:]) / 2
                distance = float(np.linalg.norm(curr - prev))
                data["movement_history"].append(distance)
                data["movement_history"] = data["movement_history"][-config.MOVEMENT_HISTORY_LENGTH:]

        # Disease classification
        models = self.app.get("models")
        if models and models.disease_models:
            for track in current_tracks:
                tid = track["id"]
                data = self.tracked_objects[tid]
                frames_tracked = frame_idx - data["creation_frame_index"]
                can_classify = (
                    (frame_idx - data["last_classified_frame"] > config.CLASSIFICATION_INTERVAL_FRAMES)
                    and (frames_tracked > config.MIN_FRAMES_FOR_CLASSIFICATION)
                    and (data.get("classification_count", 0) < config.MAX_CLASSIFICATION_COUNT)
                )
                if can_classify:
                    data["last_classified_frame"] = frame_idx
                    crop = self._get_crop(frame_orig, track["bbox_xywh"])
                    if crop is not None and crop.size > 0:
                        disease, confidence = models.classify(crop)
                        data["classification_count"] += 1
                        if disease:
                            data["disease_history"].append(disease)
                            data["disease_history"] = data["disease_history"][-config.DISEASE_HISTORY_LENGTH:]
                            logger.info(
                                f"[CLASSIFY] Track {tid}: {disease} (conf={confidence:.2f}, run={data['classification_count']})"
                            )

        # Lethargy demo target selection
        lethargy_demo = self.app.get("lethargy_demo_active", False)
        if lethargy_demo:
            if self._lethargy_demo_target_id is None and self.tracked_objects:
                self._lethargy_demo_target_id = list(self.tracked_objects.keys())[0]
                logger.info(f"[DEMO] Lethargy target selected: {self._lethargy_demo_target_id}")
            if self._lethargy_demo_target_id is not None and self._lethargy_demo_target_id not in self.tracked_objects:
                self._lethargy_demo_target_id = None
                logger.info("[DEMO] Lethargy target lost, will pick a new one.")
        else:
            self._lethargy_demo_target_id = None

        return self._draw(frame_res, current_tracks)

    def _get_crop(self, frame_orig, bbox_xywh):
        try:
            h_orig, w_orig = frame_orig.shape[:2]
            x, y, w, h = bbox_xywh
            # Scale from resized coordinates to original
            x_orig = int(x * w_orig / config.FRAME_WIDTH)
            y_orig = int(y * h_orig / config.FRAME_HEIGHT)
            w_orig_crop = int(w * w_orig / config.FRAME_WIDTH)
            h_orig_crop = int(h * h_orig / config.FRAME_HEIGHT)
            padding = 15
            x1 = max(0, x_orig - padding)
            y1 = max(0, y_orig - padding)
            x2 = min(w_orig, x_orig + w_orig_crop + padding)
            y2 = min(h_orig, y_orig + h_orig_crop + padding)
            return frame_orig[y1:y2, x1:x2]
        except Exception as e:
            logger.warning(f"Failed to get crop: {e}")
            return None

    def _draw(self, frame, tracks):
        for t in tracks:
            tid = t["id"]
            x, y, w, h = t["bbox_xywh"]
            data = self.tracked_objects[tid]
            color = (0, 255, 0)
            label = f"T:{tid}"
            thickness = 2

            is_lethargy_demo_target = self.app.get("lethargy_demo_active", False) and tid == self._lethargy_demo_target_id
            if is_lethargy_demo_target:
                color = (0, 165, 255)
                label = f"T:{tid} - Lethargic"
                thickness = 3
            elif data.get("active_alert"):
                alert = data["active_alert"]
                if alert.get("type") == "disease":
                    color = (0, 0, 255)
                    label = f"T:{tid} - {alert.get('disease', 'Disease')}!"
                elif alert.get("type") == "behavior":
                    color = (0, 165, 255)
                    label = f"T:{tid} - Lethargic"
                thickness = 3
            elif data.get("classification_count", 0) >= config.MAX_CLASSIFICATION_COUNT:
                color = (128, 128, 128)
                label = f"T:{tid} - Done"
            elif data.get("disease_history"):
                last_disease = data["disease_history"][-1]
                if last_disease != "Low Confidence":
                    color = (0, 255, 255)
                    label = f"T:{tid} - {last_disease}?"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.1, color, thickness
            )
        return frame, tracks

    def check_anomalies(self, frame_idx):
        """Check for disease and behavior alerts. Must be called while holding _lock or from process_frame."""
        with self._lock:
            return self._check_anomalies_locked(frame_idx)

    def _check_anomalies_locked(self, frame_idx):
        from collections import Counter

        disease_alerts = []
        behavior_alerts = []

        # Disease alerts
        for tid, data in self.tracked_objects.items():
            if data.get("active_alert") is None and data.get("disease_history"):
                disease_only = [d for d in data["disease_history"] if d != "Low Confidence"]
                if disease_only:
                    counts = Counter(disease_only)
                    most_common, count = counts.most_common(1)[0]
                    if count >= config.DISEASE_ALERT_THRESHOLD_COUNT:
                        alert = {
                            "type": "disease_alert",
                            "timestamp": time.time(),
                            "track_id": tid,
                            "disease": most_common,
                            "message": f"Chicken #{tid} shows consistent signs of {most_common}.",
                        }
                        disease_alerts.append(alert)
                        data["active_alert"] = {"type": "disease", "disease": most_common}
                        logger.warning(f"ALERT TRIGGERED: {alert['message']}")

        # Behavior analysis
        active_tracks = [
            data for data in self.tracked_objects.values()
            if data["frames_unseen"] == 0 and len(data.get("movement_history", [])) > 0
        ]
        if not active_tracks:
            return disease_alerts, 0.0

        flock_avg = float(np.mean([
            np.mean(d["movement_history"]) for d in active_tracks
        ]))

        for tid, data in self.tracked_objects.items():
            if data.get("active_alert") is None and len(data.get("movement_history", [])) > 0:
                individual_avg = float(np.mean(data["movement_history"]))
                is_lethargic = False

                # Demo mode bypasses flock-average check entirely
                if self.app.get("lethargy_demo_active") and tid == self._lethargy_demo_target_id:
                    is_lethargic = True
                elif flock_avg >= config.LETHARGY_FLOCK_ACTIVE_THRESHOLD:
                    is_lethargic = individual_avg < (flock_avg * config.LETHARGY_INDIVIDUAL_THRESHOLD_RATIO)

                if is_lethargic:
                    if data["lethargy_start_time"] == 0:
                        data["lethargy_start_time"] = time.time()
                    elif time.time() - data["lethargy_start_time"] > config.LETHARGY_TIMEFRAME_SECONDS:
                        alert = {
                            "type": "behavior_alert",
                            "timestamp": time.time(),
                            "track_id": tid,
                            "message": f"Warning: Chicken #{tid} is showing signs of lethargy (sustained low activity).",
                        }
                        behavior_alerts.append(alert)
                        data["active_alert"] = {"type": "behavior"}
                        logger.warning(f"ALERT TRIGGERED: {alert['message']}")
                else:
                    data["lethargy_start_time"] = 0

        return disease_alerts + behavior_alerts, flock_avg

    def get_debug_state(self):
        with self._lock:
            tracked = {}
            for tid, data in self.tracked_objects.items():
                # Include all recently seen tracks, even if they have no movement history yet
                if data.get("frames_unseen", 999) <= config.MAX_UNSEEN_FRAMES_TRACKER:
                    mh = data.get("movement_history", [])
                    tracked[tid] = {
                        "movement": float(np.mean(mh)) if mh else 0.0,
                        "dz_history": list(data.get("disease_history", [])),
                        "clf_count": data.get("classification_count", 0),
                        "frames_unseen": data.get("frames_unseen", 0),
                    }
            return tracked

    def inject_disease(self, disease_name):
        """Forcibly inject a disease into an active track to trigger immediate visual alert."""
        with self._lock:
            if not self.tracked_objects:
                return None
            # Pick the first currently visible track
            tid = None
            for k, v in self.tracked_objects.items():
                if v.get("frames_unseen", 999) == 0:
                    tid = k
                    break
            if tid is None:
                tid = list(self.tracked_objects.keys())[0]
            data = self.tracked_objects[tid]
            # Inject enough history entries to bypass the alert threshold
            for _ in range(config.DISEASE_ALERT_THRESHOLD_COUNT):
                data["disease_history"].append(disease_name)
            data["disease_history"] = data["disease_history"][-config.DISEASE_HISTORY_LENGTH:]
            data["active_alert"] = {"type": "disease", "disease": disease_name}
            return {
                "type": "disease_alert",
                "timestamp": time.time(),
                "track_id": tid,
                "disease": disease_name,
                "message": f"Chicken #{tid} shows consistent signs of {disease_name}.",
                "is_demo": True,
            }
