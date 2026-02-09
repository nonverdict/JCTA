# --- START OF FILE server_websocket2.py (FINAL VERSION) ---
import asyncio
import cv2
import json
import time
import logging
from pathlib import Path
from aiohttp import web
import concurrent.futures
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort 
from PIL import Image
from torchvision import transforms
from collections import Counter

# --- Hardware-specific Imports ---
try:
    from picamera2 import Picamera2
    IS_PICAMERA_AVAILABLE = True
except (ImportError, NotImplementedError):
    IS_PICAMERA_AVAILABLE = False

try:
    from rpi_mlx90640 import MLX90640Reader
    IS_RASPBERRY_PI = True
except (ImportError, NotImplementedError):
    IS_RASPBERRY_PI = False
    MLX90640Reader = None

from perlin_noise import PerlinNoise

# --- Configuration ---
HOST = '0.0.0.0'
PORT = 5000
WEBCAM_INDICES = [0, 2]
WEBCAM_IS_VIDEO_FILE = False
STATIC_VIDEO_PATH_STR = 'static/chicken_demo.mp4'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 70
FRAME_DELAY = 1 / 10
CAMERA_INIT_TIMEOUT = 25
STATIC_VIDEO_INIT_TIMEOUT = 15
THERMAL_FRAME_DELAY = 1 / 10

# --- YOLO Detection Configuration ---
YOLO_MODEL_PATH = 'yolov8n.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD_NMS = 0.45
YOLO_TARGET_CLASSES = [14] # COCO class for "bird"

# --- Disease Classification Configuration ---
DISEASE_MODEL_PATHS = [f'static/model_fold_{i}_quant.onnx' for i in range(1, 6)]
DISEASE_CLASS_NAMES = { 0: 'Coccidiosis', 1: 'Bootolism', 2: 'Newcastle Disease', 3: 'Chickenpox', 4: 'Lice and Mites' }
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.65
CLASSIFICATION_INTERVAL_FRAMES = 20
DISEASE_HISTORY_LENGTH = 10
DISEASE_ALERT_THRESHOLD_COUNT = 3
MIN_FRAMES_FOR_CLASSIFICATION = 15
# --- NEW --- Added from server_websocket.py to stop re-analyzing chickens
MAX_CLASSIFICATION_COUNT = 5 

# --- Behavior Analysis Configuration ---
MOVEMENT_HISTORY_LENGTH = 60
LETHARGY_TIMEFRAME_SECONDS = 30
LETHARGY_FLOCK_ACTIVE_THRESHOLD = 2.0
LETHARGY_INDIVIDUAL_THRESHOLD_RATIO = 0.2

# --- History Configuration ---
HISTORY_FILE = Path(__file__).parent / 'history.json'
history_lock = asyncio.Lock()

# --- Globals ---
STATIC_FILES_DIR = Path(__file__).parent.resolve()
yolo_model = None
DISEASE_MODELS = {} 
DISEASE_TRANSFORMS = None
tracked_objects = {}; next_track_id = 0; current_frame_index = 0
MAX_UNSEEN_FRAMES_TRACKER = 25; TRACKER_MAX_DIST = 100
CLIENTS = set(); SUBSCRIPTIONS = { "normal_video": set(), "static_video": set(), "thermal_video": set(), "debug_stream": set() }
LETHARGY_DEMO_ACTIVE = False; LETHARGY_DEMO_TARGET_ID = None
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s')
logger = logging.getLogger("PoultryScopeServer")

normal_video_source, normal_video_lock, normal_video_task = None, asyncio.Lock(), None
static_video_source, static_video_lock, static_video_task = None, asyncio.Lock(), None
thermal_camera, thermal_video_task = None, None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix='ProcWorker')
demo_image = None
CURRENT_SOURCE = "static_video"

CURRENT_CAMERA_INDEX = 0
CAMERA_SOURCES = {}
CAMERA_BROADCAST_TASKS = {}
CAMERA_TRACKING_STATES = {}

class CameraManager:
    """An abstraction layer to handle different video sources: PiCamera, USB Webcam, or Video File."""
    def __init__(self, source, is_file=False, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self._source = source
        self._is_file = is_file
        self._width = width
        self._height = height
        self._is_picam_active = False
        self._picam = None
        self._cv_cap = None
        self.source_description = "Unknown"

    def start(self):
        """Initializes and starts the camera source. Returns True on success."""
        if self._is_file:
            self.source_description = f"Video File ({self._source})"
            path_obj = Path(self._source)
            resolved_path = path_obj if path_obj.is_absolute() else (STATIC_FILES_DIR / path_obj).resolve()
            if not resolved_path.is_file():
                logger.error(f"[CameraManager] Video file not found: {resolved_path}")
                return False
            self._cv_cap = cv2.VideoCapture(str(resolved_path))
        elif WEBCAM_INDICES:
            self.source_description = f"USB Webcam (Index {self._source})"
            self._cv_cap = cv2.VideoCapture(self._source, cv2.CAP_ANY)
        elif IS_PICAMERA_AVAILABLE:
            try:
                self._picam = Picamera2()
                config = self._picam.create_video_configuration(main={"size": (self._width, self._height)})
                self._picam.configure(config)
                self._picam.start()
                self._is_picam_active = True
                self.source_description = "Raspberry Pi CSI Camera"
                logger.info(f"[CameraManager] Successfully initialized {self.source_description}.")
            except Exception as e:
                logger.error(f"[CameraManager] Failed to initialize PiCamera: {e}. Falling back to USB webcam.")
                if self._picam: self._picam.close()
                self._picam, self._is_picam_active = None, False
                self.source_description = f"USB Webcam (Index {self._source})"
                self._cv_cap = cv2.VideoCapture(self._source, cv2.CAP_ANY)
        else:
            self.source_description = f"USB Webcam (Index {self._source})"
            self._cv_cap = cv2.VideoCapture(self._source, cv2.CAP_ANY)

        if self.is_open():
            logger.info(f"[CameraManager] {self.source_description} is now open.")
            return True
        else:
            logger.error(f"[CameraManager] Failed to open {self.source_description}.")
            return False

    def is_open(self):
        """Checks if the camera source is open and ready."""
        if self._is_picam_active:
            return self._picam and self._picam.is_open
        return self._cv_cap and self._cv_cap.isOpened()

    def read(self):
        """Reads a frame from the active camera source. Returns a BGR frame or None."""
        if not self.is_open():
            return None
        try:
            if self._is_picam_active:
                frame_rgb = self._picam.capture_array()
                return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                success, frame_bgr = self._cv_cap.read()
                return frame_bgr if success else None
        except Exception as e:
            logger.error(f"[CameraManager] Error reading frame from {self.source_description}: {e}", exc_info=True)
            return None

    def release(self):
        """Stops and releases the camera source."""
        logger.info(f"[CameraManager] Releasing {self.source_description}...")
        if self._is_picam_active and self._picam:
            try:
                self._picam.stop()
                self._picam.close()
                logger.info("[CameraManager] PiCamera released.")
            except Exception as e:
                logger.error(f"[CameraManager] Error releasing PiCamera: {e}")
        if self._cv_cap:
            self._cv_cap.release()
            logger.info("[CameraManager] OpenCV capture released.")
        self._is_picam_active = False
        self._picam = None
        self._cv_cap = None

class MultiCameraManager:
    """Manages multiple USB webcams for simultaneous or selectable streaming."""
    def __init__(self, camera_indices, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self._width = width
        self._height = height
        self._camera_indices = camera_indices
        self._cameras = {}
        self._camera_names = {}
        self._available_cameras = []
        self._initialize_camera_names()

    def _initialize_camera_names(self):
        for idx in self._camera_indices:
            self._camera_names[idx] = f"USB Webcam (Index {idx})"

    def get_available_cameras(self):
        return self._available_cameras

    def start(self):
        """Initializes all available cameras. Returns True if at least one camera opens."""
        successful_cameras = 0
        for idx in self._camera_indices:
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    cap.set(cv2.CAP_PROP_FPS, 10)
                    self._cameras[idx] = cap
                    self._available_cameras.append(idx)
                    logger.info(f"[MultiCameraManager] Successfully opened {self._camera_names[idx]}")
                    successful_cameras += 1
                else:
                    cap.release()
                    logger.warning(f"[MultiCameraManager] Could not open {self._camera_names[idx]}")
            except Exception as e:
                logger.error(f"[MultiCameraManager] Error opening camera index {idx}: {e}")
        
        if successful_cameras > 0:
            logger.info(f"[MultiCameraManager] {successful_cameras}/{len(self._camera_indices)} cameras available")
            return True
        else:
            logger.error("[MultiCameraManager] No cameras could be opened")
            return False

    def is_open(self, camera_idx=None):
        if camera_idx is None:
            return len(self._cameras) > 0
        return camera_idx in self._cameras and self._cameras[camera_idx].isOpened()

    def read(self, camera_idx=None):
        target_idx = camera_idx if camera_idx is not None and camera_idx in self._cameras else self._available_cameras[0]
        if target_idx not in self._cameras:
            return None
        try:
            success, frame = self._cameras[target_idx].read()
            return frame if success else None
        except Exception as e:
            logger.error(f"[MultiCameraManager] Error reading frame from camera {target_idx}: {e}")
            return None

    def read_all(self):
        frames = {}
        for idx in self._available_cameras:
            try:
                success, frame = self._cameras[idx].read()
                frames[idx] = frame if success else None
            except Exception as e:
                logger.error(f"[MultiCameraManager] Error reading frame from camera {idx}: {e}")
                frames[idx] = None
        return frames

    def release(self):
        logger.info("[MultiCameraManager] Releasing all cameras...")
        for idx, cap in self._cameras.items():
            try:
                cap.release()
                logger.info(f"[MultiCameraManager] Released camera {idx}")
            except Exception as e:
                logger.error(f"[MultiCameraManager] Error releasing camera {idx}: {e}")
        self._cameras.clear()
        self._available_cameras.clear()

    def get_camera_name(self, idx):
        return self._camera_names.get(idx, f"Unknown Camera ({idx})")

class ThermalCameraManager:
    def __init__(self):
        self._camera = None
        self._simulation_mode = False
        self._sim_time = 0
        self._noise = None

        if IS_RASPBERRY_PI and MLX90640Reader:
            try:
                self._camera = MLX90640Reader()
                logger.info("Successfully initialized MLX90640 thermal camera.")
            except Exception as e:
                logger.error(f"Failed to initialize MLX90640 camera: {e}. Switching to simulation mode.")
                self._simulation_mode = True
                self._initialize_simulation()
        else:
            logger.warning("Not on a Raspberry Pi or adafruit_mlx90640 library not found. Using thermal camera simulation.")
            self._simulation_mode = True
            self._initialize_simulation()

    def _initialize_simulation(self):
        self._noise = PerlinNoise(octaves=2, seed=int(time.time()))

    def read_frame(self):
        if not self._simulation_mode and self._camera:
            try:
                frame = np.zeros((24 * 32,))
                self._camera.mlx.getFrame(frame)
                return frame.reshape((24, 32))
            except Exception as e:
                logger.error(f"Error reading from MLX90640: {e}")
                return None
        else:
            self._sim_time += 0.05
            frame = np.zeros((24, 32))
            for y in range(24):
                for x in range(32):
                    frame[y][x] = self._noise([x / 32, y / 24, self._sim_time])
            
            frame = 34 + (frame * 4)
            np.clip(frame, 30, 38, out=frame)
            return frame

    def close(self):
        if self._camera:
            self._camera.close()

# (ONNX Disease Classification functions remain unchanged)
def get_onnx_session(model_path):
    try:
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), providers=providers)
        return session
    except Exception as e:
        logger.error(f"[Executor] Failed to load ONNX model {model_path}: {e}", exc_info=True)
        return None

async def initialize_disease_models_async():
    global DISEASE_MODELS, DISEASE_TRANSFORMS
    if DISEASE_MODELS: return True
    logger.info("Disease classification will run on CPU using ONNX Runtime.")
    DISEASE_TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loop = asyncio.get_running_loop()
    logger.info("Initializing ONNX disease classification models...")
    models_loaded = 0
    for path_str in DISEASE_MODEL_PATHS:
        model_path = STATIC_FILES_DIR / path_str
        if not model_path.is_file():
            logger.warning(f"ONNX model file not found: {model_path}. Skipping.")
            continue
        session = await loop.run_in_executor(executor, get_onnx_session, model_path)
        if session:
            DISEASE_MODELS[path_str] = session
            models_loaded += 1
    if models_loaded > 0:
        logger.info(f"Successfully loaded {models_loaded}/{len(DISEASE_MODEL_PATHS)} ONNX models.")
        return True
    else:
        logger.error("No ONNX disease models loaded. This feature will be disabled."); return False

def classify_chicken_ensemble_blocking(chicken_crop_np, models, transform):
    if not models: return None, 0.0
    try:
        img = Image.fromarray(cv2.cvtColor(chicken_crop_np, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0)
        ort_inputs = { 'input': img_tensor.numpy() }
    except Exception as e:
        logger.error(f"[Executor] Failed to transform chicken crop for ONNX: {e}"); return None, 0.0

    best_confidence = -1.0
    best_class_name = "Low Confidence"

    for model_name, session in models.items():
        try:
            ort_outs = session.run(None, ort_inputs)
            output_np = ort_outs[0]
            exp_scores = np.exp(output_np - np.max(output_np))
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            confidence = np.max(probabilities)
            predicted_class_idx = np.argmax(probabilities)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_class_name = DISEASE_CLASS_NAMES.get(predicted_class_idx.item(), "Unknown")

        except Exception as e:
            logger.error(f"[Executor] ONNX Inference error with model {model_name}: {e}"); continue
    
    if best_confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:
        return "Low Confidence", best_confidence
    else:
        return best_class_name, best_confidence

# (YOLO and stream initialization remain unchanged)
async def initialize_yolo_model_async():
    global yolo_model
    if yolo_model is not None: return True
    logger.info(f"Initializing YOLO model using '{YOLO_MODEL_PATH}'...")
    try:
        loop = asyncio.get_running_loop()
        model_path_obj = Path(YOLO_MODEL_PATH)
        true_model_source = str(model_path_obj.resolve()) if model_path_obj.is_file() else YOLO_MODEL_PATH
        logger.info(f"Attempting to load YOLO model from: {true_model_source}")
        temp_model = await loop.run_in_executor(executor, lambda: YOLO(true_model_source))
        logger.warning("Forcing YOLO to run on CPU for Raspberry Pi.")
        temp_model.to('cpu')
        yolo_model = temp_model
        logger.info(f"YOLO model loaded. Device: {yolo_model.device}")
        dummy_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        logger.info("Performing YOLO model warmup...")
        await loop.run_in_executor(executor, lambda: yolo_model(dummy_frame, verbose=False, conf=0.05))
        logger.info("YOLO model warmup complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model: {e}", exc_info=True); yolo_model = None; return False

async def initialize_stream_source(stream_type: str):
    global normal_video_source, static_video_source
    if not await initialize_yolo_model_async():
        logger.error(f"Cannot initialize {stream_type} source: YOLO model failed.")
        return False

    is_static = stream_type == "static_video"
    lock = static_video_lock if is_static else normal_video_lock

    async with lock:
        current_cam_manager = static_video_source if is_static else normal_video_source
        if current_cam_manager and current_cam_manager.is_open():
            logger.debug(f"{stream_type.capitalize()} source already initialized.")
            return True

        source_id_or_path = STATIC_VIDEO_PATH_STR if is_static else WEBCAM_INDICES[0]
        is_file = is_static or (not is_static and WEBCAM_IS_VIDEO_FILE)
        timeout = STATIC_VIDEO_INIT_TIMEOUT if is_file else CAMERA_INIT_TIMEOUT

        cam_manager = CameraManager(source=source_id_or_path, is_file=is_file)

        logger.info(f"Requesting {stream_type} source initialization (timeout: {timeout}s)...")
        loop = asyncio.get_running_loop()
        try:
            success = await asyncio.wait_for(loop.run_in_executor(executor, cam_manager.start), timeout=timeout)
            if success:
                if is_static:
                    static_video_source = cam_manager
                else:
                    normal_video_source = cam_manager
                logger.info(f"Source initialization successful for {cam_manager.source_description}.")
                return True
            else:
                logger.error(f"Source initialization failed for {stream_type}.")
                return False
        except asyncio.TimeoutError:
            logger.error(f"Source initialization timed out for {stream_type}.")
            cam_manager.release() 
            return False
        except Exception as e:
            logger.error(f"Unexpected error during {stream_type} init: {e}", exc_info=True)
            cam_manager.release()
            return False

async def release_stream_source(stream_type: str):
    global normal_video_source, normal_video_task, static_video_source, static_video_task
    is_static = stream_type == "static_video"
    lock = static_video_lock if is_static else normal_video_lock

    async with lock:
        cam_manager_to_release = static_video_source if is_static else normal_video_source
        task_to_cancel = static_video_task if is_static else normal_video_task

        if is_static:
            static_video_source, static_video_task = None, None
        else:
            normal_video_source, normal_video_task = None, None

        if task_to_cancel and not task_to_cancel.done():
            desc = cam_manager_to_release.source_description if cam_manager_to_release else stream_type
            logger.info(f"Cancelling {desc} broadcast task...")
            task_to_cancel.cancel()
            try:
                await task_to_cancel
            except asyncio.CancelledError:
                logger.info(f"{desc} task cancelled.")

        if cam_manager_to_release:
            logger.info(f"Requesting {cam_manager_to_release.source_description} release via executor...")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(executor, cam_manager_to_release.release)

        logger.info(f"{stream_type.capitalize()} source release process finished.")
        
# --- Main CV processing function ---
def _process_frame_yolo_tracking_blocking(frame_bgr_original, frame_bgr_resized, frame_idx):
    global yolo_model, tracked_objects, next_track_id, DISEASE_MODELS, DISEASE_TRANSFORMS
    if yolo_model is None: return frame_bgr_resized, []
    processed_frame = frame_bgr_resized.copy(); detections_for_tracker = []
    try:
        results = yolo_model(processed_frame, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD, iou=YOLO_IOU_THRESHOLD_NMS, classes=YOLO_TARGET_CLASSES)
    except Exception as e:
        logger.error(f"Error during YOLO inference: {e}", exc_info=True); return frame_bgr_resized, []
    if results and len(results) > 0:
        res = results[0]
        for i in range(len(res.boxes.xyxy)):
            x1, y1, x2, y2 = map(int, res.boxes.xyxy[i]); w, h = x2 - x1, y2 - y1; cx, cy = x1 + w // 2, y1 + h // 2
            detections_for_tracker.append({'bbox_center': (cx, cy), 'bbox_xywh': (x1, y1, w, h)})
            
    current_tracks_output = []; temp_tracked_objects_next_state = {}; unmatched_detection_indices = list(range(len(detections_for_tracker)))
    
    # Match existing tracks
    for track_id, data in tracked_objects.items():
        data["frames_unseen"] += 1; best_match_idx, min_dist = -1, TRACKER_MAX_DIST
        for det_idx in unmatched_detection_indices:
            dist = np.linalg.norm(np.array(data["bbox_center"]) - np.array(detections_for_tracker[det_idx]['bbox_center']))
            if dist < min_dist: min_dist, best_match_idx = dist, det_idx
        if best_match_idx != -1:
            matched_det = detections_for_tracker[best_match_idx]
            data.update({"bbox_center": matched_det['bbox_center'], "last_seen_frame": frame_idx, "frames_unseen": 0, "bbox_history": (data["bbox_history"] + [matched_det['bbox_xywh']])[-10:]})
            current_tracks_output.append({"id": track_id, "bbox_xywh": matched_det['bbox_xywh']}); unmatched_detection_indices.remove(best_match_idx); temp_tracked_objects_next_state[track_id] = data
        elif data["frames_unseen"] <= MAX_UNSEEN_FRAMES_TRACKER:
            temp_tracked_objects_next_state[track_id] = data
            
    # Create new tracks
    for det_idx in unmatched_detection_indices:
        new_det = detections_for_tracker[det_idx]
        temp_tracked_objects_next_state[next_track_id] = {
            "bbox_center": new_det['bbox_center'], "creation_frame_index": frame_idx, "last_seen_frame": frame_idx, "frames_unseen": 0,
            "bbox_history": [new_det['bbox_xywh']], "disease_history": [], "last_classified_frame": 0, "active_alert": None,
            "movement_history": [], "lethargy_start_time": 0,
            # --- NEW --- Initialize classification count for the new track
            "classification_count": 0
        }
        current_tracks_output.append({"id": next_track_id, "bbox_xywh": new_det['bbox_xywh']}); next_track_id = (next_track_id + 1) % 20000
    tracked_objects.clear(); tracked_objects.update(temp_tracked_objects_next_state)

    # Calculate Movement
    for track in current_tracks_output:
        track_id = track["id"]
        data = tracked_objects[track_id]
        if len(data["bbox_history"]) > 1:
            prev_center = np.array(data["bbox_history"][-2][:2]) + np.array(data["bbox_history"][-2][2:]) / 2
            curr_center = np.array(data["bbox_history"][-1][:2]) + np.array(data["bbox_history"][-1][2:]) / 2
            distance = np.linalg.norm(curr_center - prev_center)
            data["movement_history"].append(distance)
            data["movement_history"] = data["movement_history"][-MOVEMENT_HISTORY_LENGTH:]

    # Conditional Disease Classification with Limiting
    if DISEASE_MODELS:
        for track in current_tracks_output:
            track_id = track["id"]
            data = tracked_objects[track_id]
            frames_tracked = data["last_seen_frame"] - data["creation_frame_index"]

            # --- NEW --- The complete, combined condition
            if (frame_idx - data["last_classified_frame"] > CLASSIFICATION_INTERVAL_FRAMES) and \
               (frames_tracked > MIN_FRAMES_FOR_CLASSIFICATION) and \
               (data.get("classification_count", 0) < MAX_CLASSIFICATION_COUNT):

                data["last_classified_frame"] = frame_idx
                h_orig, w_orig, _ = frame_bgr_original.shape; h_res, w_res, _ = frame_bgr_resized.shape
                x, y, w, h = track["bbox_xywh"]
                x_orig, y_orig = int(x * w_orig / w_res), int(y * h_orig / h_res); w_orig, h_orig = int(w * w_orig / w_res), int(h * h_orig / h_res)
                padding = 15; crop_x1, crop_y1 = max(0, x_orig - padding), max(0, y_orig - padding); crop_x2, crop_y2 = min(frame_bgr_original.shape[1], x_orig + w_orig + padding), min(frame_bgr_original.shape[0], y_orig + h_orig + padding)
                chicken_crop = frame_bgr_original[crop_y1:crop_y2, crop_x1:crop_x2]
                
                if chicken_crop.size > 0:
                    disease, confidence = classify_chicken_ensemble_blocking(chicken_crop, DISEASE_MODELS, DISEASE_TRANSFORMS)
                    # --- NEW --- Increment counter after each attempt
                    data["classification_count"] = data.get("classification_count", 0) + 1
                    logger.info(f"[ONNX ENSEMBLE] Track {track_id}: Result='{disease}', Conf={confidence:.2f}, Run #{data['classification_count']}")
                    if disease:
                        history = data["disease_history"]; history.append(disease); data["disease_history"] = history[-DISEASE_HISTORY_LENGTH:]

    # Lethargy Demo Logic
    global LETHARGY_DEMO_ACTIVE, LETHARGY_DEMO_TARGET_ID
    if LETHARGY_DEMO_ACTIVE and LETHARGY_DEMO_TARGET_ID is None and tracked_objects:
        LETHARGY_DEMO_TARGET_ID = list(tracked_objects.keys())[0]
        logger.info(f"[DEMO] Lethargy target selected: {LETHARGY_DEMO_TARGET_ID}")
    if LETHARGY_DEMO_ACTIVE and LETHARGY_DEMO_TARGET_ID is not None and LETHARGY_DEMO_TARGET_ID not in tracked_objects:
        LETHARGY_DEMO_TARGET_ID = None
        logger.info("[DEMO] Lethargy target lost, will pick a new one.")

    # Visualization
    for track in current_tracks_output:
        x, y, w, h = track["bbox_xywh"]; track_id = track["id"]; data = tracked_objects[track_id]
        color = (0, 255, 0); thickness = 2; label = f"T:{track_id}"
        
        is_lethargy_demo_target = LETHARGY_DEMO_ACTIVE and track_id == LETHARGY_DEMO_TARGET_ID

        if is_lethargy_demo_target:
            color = (0, 165, 255); label = f"T:{track_id} - Lethargic"; thickness = 3
        elif data.get("active_alert"):
            if data["active_alert"]["type"] == "disease":
                color = (0, 0, 255); label = f"T:{track_id} - {data['active_alert']['disease']}!"
            elif data["active_alert"]["type"] == "behavior":
                color = (0, 165, 255); label = f"T:{track_id} - Lethargic"
            thickness = 3
        # --- NEW --- Visualization for "Done" chickens
        elif data.get("classification_count", 0) >= MAX_CLASSIFICATION_COUNT:
            color = (128, 128, 128) # Gray
            label = f"T:{track_id} - Done"
        elif data["disease_history"]:
            last_disease = data["disease_history"][-1]
            if last_disease != "Low Confidence":
                color = (0, 255, 255); label = f"T:{track_id} - {last_disease}?"
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, thickness); cv2.putText(processed_frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.1, color, thickness)
    return processed_frame, current_tracks_output

# (The rest of the file: analyze_anomalies, add_alert_to_history, broadcast loops, 
# websocket handlers, and main function remain unchanged.)

async def analyze_anomalies(frame_idx):
    """Combined analysis for both disease and behavior."""
    disease_alerts = []
    behavior_alerts = []

    # 1. Disease Analysis
    if DISEASE_MODELS:
        for track_id, data in tracked_objects.items():
            if data.get("active_alert") is None and data["disease_history"]:
                disease_only_history = [d for d in data["disease_history"] if d != "Low Confidence"]
                if disease_only_history:
                    disease_counts = Counter(disease_only_history)
                    most_common_disease, count = disease_counts.most_common(1)[0]
                    if count >= DISEASE_ALERT_THRESHOLD_COUNT:
                        alert = {"type": "disease_alert", "timestamp": time.time(), "track_id": track_id, "disease": most_common_disease, "message": f"Chicken #{track_id} shows consistent signs of {most_common_disease}."}
                        disease_alerts.append(alert)
                        tracked_objects[track_id]["active_alert"] = {"type": "disease", "disease": most_common_disease}
                        logger.warning(f"ALERT TRIGGERED: {alert['message']}")

    # 2. Behavior Analysis
    active_tracks_with_movement = [data for data in tracked_objects.values() if data["frames_unseen"] == 0 and len(data["movement_history"]) > 0]
    if not active_tracks_with_movement:
        all_alerts = disease_alerts
        if all_alerts:
            for alert in all_alerts:
                asyncio.create_task(add_alert_to_history(alert))
        return all_alerts, 0

    flock_avg_movement = np.mean([np.mean(data["movement_history"]) for data in active_tracks_with_movement])

    if flock_avg_movement < LETHARGY_FLOCK_ACTIVE_THRESHOLD:
        all_alerts = disease_alerts
        if all_alerts:
            for alert in all_alerts:
                asyncio.create_task(add_alert_to_history(alert))
        return all_alerts, flock_avg_movement 

    for track_id, data in tracked_objects.items():
        if data.get("active_alert") is None and len(data["movement_history"]) > 0:
            individual_avg_movement = np.mean(data["movement_history"])
            is_lethargic = individual_avg_movement < (flock_avg_movement * LETHARGY_INDIVIDUAL_THRESHOLD_RATIO)

            if is_lethargic:
                if data["lethargy_start_time"] == 0:
                    data["lethargy_start_time"] = time.time()
                elif time.time() - data["lethargy_start_time"] > LETHARGY_TIMEFRAME_SECONDS:
                    alert = {"type": "behavior_alert", "timestamp": time.time(), "track_id": track_id, "message": f"Warning: Chicken #{track_id} is showing signs of lethargy (sustained low activity)."}
                    behavior_alerts.append(alert)
                    tracked_objects[track_id]["active_alert"] = {"type": "behavior"}
                    logger.warning(f"ALERT TRIGGERED: {alert['message']}")
            else:
                data["lethargy_start_time"] = 0 

    all_alerts = disease_alerts + behavior_alerts
    if all_alerts:
        for alert in all_alerts:
            asyncio.create_task(add_alert_to_history(alert))

    return all_alerts, flock_avg_movement

async def add_alert_to_history(alert_data):
    """Appends a single alert to the history file and broadcasts it to all clients."""
    async with history_lock:
        try:
            if HISTORY_FILE.is_file() and HISTORY_FILE.stat().st_size > 0:
                history = json.loads(HISTORY_FILE.read_text())
            else:
                history = []
            history.insert(0, alert_data)
            HISTORY_FILE.write_text(json.dumps(history, indent=4))
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading/writing history file: {e}")
            return

    payload = json.dumps(alert_data)
    for client_ws in CLIENTS:
        if not client_ws.closed:
            try:
                await client_ws.send_str(payload)
            except Exception as e:
                logger.warning(f"Failed to send new alert to {getattr(client_ws, 'addr_str', 'unknown')}: {e}")

async def broadcast_frames_loop(stream_type: str):
    global current_frame_index, tracked_objects, next_track_id, normal_video_source, static_video_source, demo_image, CURRENT_SOURCE
    is_static = stream_type == "static_video"
    is_file_feed = is_static or (not is_static and WEBCAM_IS_VIDEO_FILE)

    cam_manager = static_video_source if is_static else normal_video_source
    source_name = cam_manager.source_description if cam_manager else "Unknown Source"

    logger.info(f"{source_name} broadcast task started.")
    loop = asyncio.get_running_loop()

    if is_file_feed:
        tracked_objects.clear()
        next_track_id = 0

    while True:
        start_time_cycle = time.monotonic()

        cam_manager = static_video_source if is_static else normal_video_source
        is_demo_mode = stream_type == "static_video" and CURRENT_SOURCE == "demo_image"

        subscription_key = "static_video" if is_static else "normal_video"
        if not SUBSCRIPTIONS[subscription_key]:
            logger.info(f"No subscribers for {source_name}, stopping task."); break

        if (not cam_manager or not cam_manager.is_open()) and not is_demo_mode:
            logger.warning(f"{source_name} is not open. Waiting...")
            await asyncio.sleep(0.5); continue

        if yolo_model is None:
            await asyncio.sleep(0.2); continue

        try:
            if is_demo_mode:
                original_frame = demo_image.copy()
            else:
                original_frame = await loop.run_in_executor(executor, cam_manager.read)

            if original_frame is None:
                if is_file_feed:
                    logger.info(f"End of {source_name}. Looping...")
                    await release_stream_source(stream_type=stream_type)
                    if await initialize_stream_source(stream_type=stream_type):
                        cam_manager = static_video_source if is_static else normal_video_source
                        source_name = cam_manager.source_description if cam_manager else "Unknown Source"
                        tracked_objects.clear(); next_track_id = 0; continue
                    else:
                        logger.error(f"Failed to re-initialize {source_name}. Stopping task."); break
                else:
                    logger.warning(f"Failed to capture frame from {source_name}. Trying again..."); await asyncio.sleep(0.5); continue

            current_frame_index += 1
            resized_frame = cv2.resize(original_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            processed_frame, tracks = await loop.run_in_executor(executor, _process_frame_yolo_tracking_blocking, original_frame, resized_frame, current_frame_index)
            
            _, flock_avg_movement = await analyze_anomalies(current_frame_index)

            if SUBSCRIPTIONS["debug_stream"]:
                debug_data = {
                    "type": "debug_update",
                    "flock_avg_movement": flock_avg_movement,
                    "tracked_objects": {tid: {"movement": np.mean(d["movement_history"]), "dz_history": d["disease_history"], "clf_count": d.get("classification_count", 0)} for tid, d in tracked_objects.items() if d.get("movement_history")}
                }
                asyncio.create_task(send_debug_data(debug_data))

            ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ret: continue
            frame_bytes = buffer.tobytes()
            message_to_send = b'\x01' + frame_bytes
            active_subscribers = list(SUBSCRIPTIONS[subscription_key])
            if active_subscribers:
                send_tasks = [asyncio.create_task(safe_send_binary(ws, message_to_send)) for ws in active_subscribers]
                if send_tasks: await asyncio.gather(*send_tasks, return_exceptions=True)
            elapsed_cycle = time.monotonic() - start_time_cycle
            await asyncio.sleep(max(0, FRAME_DELAY - elapsed_cycle))
        except asyncio.CancelledError:
            logger.info(f"{source_name} broadcast task cancelled."); break
        except Exception as e:
            logger.error(f"Error in {source_name} broadcast loop: {e}", exc_info=True); await asyncio.sleep(1)
    logger.info(f"{source_name} broadcast task finished.")

async def broadcast_thermal_frames_loop():
    """Dedicated broadcast loop for the thermal camera stream."""
    global thermal_camera
    logger.info("Thermal camera broadcast task started.")
    while True:
        start_time_cycle = time.monotonic()
        if not SUBSCRIPTIONS["thermal_video"]:
            logger.info("No subscribers for thermal video, stopping task."); break
        try:
            temp_frame = await asyncio.get_running_loop().run_in_executor(executor, thermal_camera.read_frame)
            if temp_frame is None:
                await asyncio.sleep(1); continue
            norm_frame = cv2.normalize(temp_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            color_frame = cv2.applyColorMap(norm_frame, cv2.COLORMAP_INFERNO)
            resized_frame = cv2.resize(color_frame, (320, 240), interpolation=cv2.INTER_NEAREST)
            ret, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret: continue
            frame_bytes = buffer.tobytes()
            message_to_send = b'\x02' + frame_bytes
            active_subscribers = list(SUBSCRIPTIONS["thermal_video"])
            if active_subscribers:
                send_tasks = [asyncio.create_task(safe_send_binary(ws, message_to_send)) for ws in active_subscribers]
                if send_tasks: await asyncio.gather(*send_tasks, return_exceptions=True)
            elapsed_cycle = time.monotonic() - start_time_cycle
            await asyncio.sleep(max(0, THERMAL_FRAME_DELAY - elapsed_cycle))
        except asyncio.CancelledError:
            logger.info("Thermal broadcast task cancelled."); break
        except Exception as e:
            logger.error(f"Error in thermal broadcast loop: {e}", exc_info=True); await asyncio.sleep(1)
    logger.info("Thermal broadcast task finished.")

async def send_debug_data(data):
    payload = json.dumps(data)
    for client_ws in SUBSCRIPTIONS["debug_stream"]:
        if not client_ws.closed:
            try: await client_ws.send_str(payload)
            except Exception: pass

async def check_and_manage_normal_video_task():
    global normal_video_task, normal_video_source
    if SUBSCRIPTIONS["normal_video"]:
        if CAMERA_SOURCES and CAMERA_SOURCES.get_available_cameras():
            available_cams = CAMERA_SOURCES.get_available_cameras()
            logger.info(f"Using multicam sources for normal_video. Available cameras: {available_cams}")
            if normal_video_source is None:
                cam_idx = available_cams[0]
                normal_video_source = CameraManager(source=cam_idx, is_file=False)
                normal_video_source._cv_cap = CAMERA_SOURCES._cameras[cam_idx]
                normal_video_source.source_description = CAMERA_SOURCES.get_camera_name(cam_idx)
                logger.info(f"Reusing camera {cam_idx} from multicam sources for normal_video")
            if normal_video_task is None or normal_video_task.done():
                logger.info(f"{normal_video_source.source_description} source ready. Starting broadcast...")
                normal_video_task = asyncio.create_task(broadcast_frames_loop(stream_type="normal_video"))
        else:
            if (normal_video_source is None or not normal_video_source.is_open()) and not await initialize_stream_source(stream_type="normal_video"):
                return
            if normal_video_source is not None and (normal_video_task is None or normal_video_task.done()):
                logger.info(f"{normal_video_source.source_description} source ready. Starting broadcast...")
                normal_video_task = asyncio.create_task(broadcast_frames_loop(stream_type="normal_video"))
    elif normal_video_source is not None or (normal_video_task and not normal_video_task.done()):
        await release_stream_source(stream_type="normal_video")

async def check_and_manage_static_video_task():
    global static_video_task, static_video_source
    if SUBSCRIPTIONS["static_video"]:
        if (static_video_source is None or not static_video_source.is_open()) and not await initialize_stream_source(stream_type="static_video"):
            return
        if static_video_source is not None and (static_video_task is None or static_video_task.done()):
            logger.info(f"{static_video_source.source_description} source ready. Starting broadcast...")
            static_video_task = asyncio.create_task(broadcast_frames_loop(stream_type="static_video"))
    elif static_video_source is not None or (static_video_task and not static_video_task.done()):
        await release_stream_source(stream_type="static_video")

async def check_and_manage_thermal_video_task():
    global thermal_video_task
    if SUBSCRIPTIONS["thermal_video"]:
        if thermal_video_task is None or thermal_video_task.done():
            logger.info("Subscribers detected for thermal stream. Starting broadcast...")
            thermal_video_task = asyncio.create_task(broadcast_thermal_frames_loop())
    elif thermal_video_task and not thermal_video_task.done():
        logger.info("No more subscribers for thermal stream. Stopping broadcast...")
        thermal_video_task.cancel()

async def initialize_multicam():
    global CAMERA_SOURCES
    if not await initialize_yolo_model_async():
        logger.error("Cannot initialize multicam: YOLO model failed.")
        return False
    if CAMERA_SOURCES:
        logger.info("Multicam already initialized")
        return True
    logger.info(f"Initializing multicam with indices: {WEBCAM_INDICES}")
    loop = asyncio.get_running_loop()
    multi_cam = MultiCameraManager(WEBCAM_INDICES)
    try:
        success = await asyncio.wait_for(loop.run_in_executor(executor, multi_cam.start), timeout=CAMERA_INIT_TIMEOUT)
        if success:
            CAMERA_SOURCES = multi_cam
            available = multi_cam.get_available_cameras()
            logger.info(f"Multicam initialized successfully. Available cameras: {available}")
            return True
        else:
            logger.error("Failed to initialize multicam")
            return False
    except asyncio.TimeoutError:
        logger.error("Multicam initialization timed out")
        multi_cam.release()
        return False
    except Exception as e:
        logger.error(f"Unexpected error during multicam init: {e}", exc_info=True)
        multi_cam.release()
        return False

async def release_multicam():
    global CAMERA_SOURCES, CAMERA_BROADCAST_TASKS, CAMERA_TRACKING_STATES
    tasks_to_cancel = [t for t in CAMERA_BROADCAST_TASKS.values() if t and not t.done()]
    for task in tasks_to_cancel:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    CAMERA_BROADCAST_TASKS.clear()
    CAMERA_TRACKING_STATES.clear()
    if CAMERA_SOURCES:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, CAMERA_SOURCES.release)
        CAMERA_SOURCES = None
    logger.info("Multicam released")

def get_tracking_state(camera_idx):
    if camera_idx not in CAMERA_TRACKING_STATES:
        CAMERA_TRACKING_STATES[camera_idx] = {
            "tracked_objects": {},
            "next_track_id": 0,
            "current_frame_index": 0
        }
    return CAMERA_TRACKING_STATES[camera_idx]

async def broadcast_multicam_frames_loop(camera_idx):
    global CURRENT_SOURCE, LETHARGY_DEMO_ACTIVE, LETHARGY_DEMO_TARGET_ID
    if not CAMERA_SOURCES or not CAMERA_SOURCES.is_open(camera_idx):
        logger.warning(f"Camera {camera_idx} is not available. Stopping broadcast.")
        return
    cam_name = CAMERA_SOURCES.get_camera_name(camera_idx)
    logger.info(f"Multicam broadcast started for {cam_name}")
    loop = asyncio.get_running_loop()
    tracking_state = get_tracking_state(camera_idx)
    tracked_objects = tracking_state["tracked_objects"]
    next_track_id = tracking_state["next_track_id"]
    current_frame_index = tracking_state["current_frame_index"]
    subscription_key = f"multicam_{camera_idx}"
    if subscription_key not in SUBSCRIPTIONS:
        SUBSCRIPTIONS[subscription_key] = set()
    while True:
        start_time_cycle = time.monotonic()
        if not SUBSCRIPTIONS[subscription_key]:
            logger.info(f"No subscribers for {cam_name}, stopping task.")
            break
        if not CAMERA_SOURCES.is_open(camera_idx):
            logger.warning(f"{cam_name} is not open. Waiting...")
            await asyncio.sleep(0.5)
            continue
        if yolo_model is None:
            await asyncio.sleep(0.2)
            continue
        try:
            original_frame = await loop.run_in_executor(executor, lambda: CAMERA_SOURCES.read(camera_idx))
            if original_frame is None:
                logger.warning(f"Failed to capture frame from {cam_name}. Trying again...")
                await asyncio.sleep(0.5)
                continue
            current_frame_index += 1
            tracking_state["current_frame_index"] = current_frame_index
            resized_frame = cv2.resize(original_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            processed_frame, tracks = await loop.run_in_executor(
                executor, 
                lambda f, r, idx: _process_frame_yolo_tracking_blocking(f, r, idx),
                original_frame, resized_frame, current_frame_index
            )
            _, flock_avg_movement = await analyze_anomalies(current_frame_index)
            if SUBSCRIPTIONS["debug_stream"]:
                debug_data = {
                    "type": "debug_update",
                    "camera_idx": camera_idx,
                    "flock_avg_movement": flock_avg_movement,
                    "tracked_objects": {tid: {"movement": np.mean(d["movement_history"]), "dz_history": d["disease_history"], "clf_count": d.get("classification_count", 0)} for tid, d in tracked_objects.items() if d.get("movement_history")}
                }
                asyncio.create_task(send_debug_data(debug_data))
            ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            message_to_send = b'\x05' + bytes([camera_idx]) + frame_bytes
            active_subscribers = list(SUBSCRIPTIONS[subscription_key])
            if active_subscribers:
                send_tasks = [asyncio.create_task(safe_send_binary(ws, message_to_send)) for ws in active_subscribers]
                if send_tasks:
                    await asyncio.gather(*send_tasks, return_exceptions=True)
            elapsed_cycle = time.monotonic() - start_time_cycle
            await asyncio.sleep(max(0, FRAME_DELAY - elapsed_cycle))
        except asyncio.CancelledError:
            logger.info(f"{cam_name} broadcast task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in {cam_name} broadcast loop: {e}", exc_info=True)
            await asyncio.sleep(1)
    logger.info(f"{cam_name} broadcast task finished.")

async def check_and_manage_multicam_tasks():
    global CAMERA_BROADCAST_TASKS, CURRENT_CAMERA_INDEX
    available_cameras = CAMERA_SOURCES.get_available_cameras() if CAMERA_SOURCES else []
    for camera_idx in available_cameras:
        subscription_key = f"multicam_{camera_idx}"
        if SUBSCRIPTIONS[subscription_key]:
            if CAMERA_SOURCES and CAMERA_SOURCES.is_open(camera_idx):
                if camera_idx not in CAMERA_BROADCAST_TASKS or CAMERA_BROADCAST_TASKS[camera_idx].done():
                    cam_name = CAMERA_SOURCES.get_camera_name(camera_idx)
                    logger.info(f"{cam_name} source ready. Starting broadcast...")
                    CAMERA_BROADCAST_TASKS[camera_idx] = asyncio.create_task(broadcast_multicam_frames_loop(camera_idx))
    for camera_idx in list(CAMERA_BROADCAST_TASKS.keys()):
        subscription_key = f"multicam_{camera_idx}"
        if not SUBSCRIPTIONS[subscription_key] and CAMERA_BROADCAST_TASKS[camera_idx] and not CAMERA_BROADCAST_TASKS[camera_idx].done():
            CAMERA_BROADCAST_TASKS[camera_idx].cancel()
            try:
                await CAMERA_BROADCAST_TASKS[camera_idx]
            except asyncio.CancelledError:
                pass
            del CAMERA_BROADCAST_TASKS[camera_idx]

MULTICAM_COMBINED_TASK = None

async def broadcast_multicam_combined_loop():
    """Broadcasts all cameras in a combined grid view."""
    global MULTICAM_COMBINED_TASK
    if not CAMERA_SOURCES or not CAMERA_SOURCES.get_available_cameras():
        logger.warning("No cameras available for combined view")
        return
    
    logger.info("Multicam combined view broadcast started")
    loop = asyncio.get_running_loop()
    subscription_key = "multicam_combined"
    
    while True:
        start_time_cycle = time.monotonic()
        if not SUBSCRIPTIONS[subscription_key]:
            logger.info("No subscribers for combined view, stopping task.")
            break
        if not CAMERA_SOURCES.is_open():
            logger.warning("Cameras not available for combined view. Waiting...")
            await asyncio.sleep(0.5)
            continue
        if yolo_model is None:
            await asyncio.sleep(0.2)
            continue
        
        try:
            all_frames = CAMERA_SOURCES.read_all()
            available_cams = CAMERA_SOURCES.get_available_cameras()
            
            if not available_cams:
                await asyncio.sleep(0.5)
                continue
            
            processed_frames = []
            for cam_idx in available_cams:
                frame = all_frames.get(cam_idx)
                if frame is not None:
                    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
                    tracking_state = get_tracking_state(cam_idx)
                    processed, _ = await loop.run_in_executor(
                        executor,
                        lambda f, idx: _process_frame_yolo_tracking_blocking(f, f, tracking_state["current_frame_index"]),
                        resized
                    )
                    cam_name = CAMERA_SOURCES.get_camera_name(cam_idx)
                    cv2.putText(processed, cam_name, (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
                    processed_frames.append(processed)
                else:
                    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"Camera {cam_idx}: No Signal", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
                    processed_frames.append(placeholder)
            
            if len(processed_frames) == 2:
                combined = cv2.hconcat(processed_frames)
            elif len(processed_frames) == 1:
                combined = processed_frames[0]
            elif len(processed_frames) >= 2:
                row1 = cv2.hconcat(processed_frames[:2])
                if len(processed_frames) > 2:
                    row2 = cv2.hconcat(processed_frames[2:4])
                    combined = cv2.vconcat([row1, row2])
                else:
                    combined = row1
            else:
                combined = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(combined, "No Cameras Available", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            message_to_send = b'\x06' + frame_bytes
            active_subscribers = list(SUBSCRIPTIONS[subscription_key])
            if active_subscribers:
                send_tasks = [asyncio.create_task(safe_send_binary(ws, message_to_send)) for ws in active_subscribers]
                if send_tasks:
                    await asyncio.gather(*send_tasks, return_exceptions=True)
            
            elapsed_cycle = time.monotonic() - start_time_cycle
            await asyncio.sleep(max(0, FRAME_DELAY - elapsed_cycle))
        except asyncio.CancelledError:
            logger.info("Multicam combined view broadcast cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in combined view broadcast loop: {e}", exc_info=True)
            await asyncio.sleep(1)
    
    logger.info("Multicam combined view broadcast finished.")

async def check_and_manage_multicam_combined_task():
    global MULTICAM_COMBINED_TASK
    subscription_key = "multicam_combined"
    if SUBSCRIPTIONS[subscription_key]:
        if MULTICAM_COMBINED_TASK is None or MULTICAM_COMBINED_TASK.done():
            logger.info("Subscribers detected for combined view. Starting broadcast...")
            MULTICAM_COMBINED_TASK = asyncio.create_task(broadcast_multicam_combined_loop())
    elif MULTICAM_COMBINED_TASK and not MULTICAM_COMBINED_TASK.done():
        logger.info("No more subscribers for combined view. Stopping broadcast...")
        MULTICAM_COMBINED_TASK.cancel()
        try:
            await MULTICAM_COMBINED_TASK
        except asyncio.CancelledError:
            pass
        MULTICAM_COMBINED_TASK = None

async def handle_websocket_message(ws, addr, message_str):
    global CURRENT_SOURCE, LETHARGY_DEMO_ACTIVE, LETHARGY_DEMO_TARGET_ID, CURRENT_CAMERA_INDEX
    try:
        data = json.loads(message_str)
        action = data.get("action")
        
        if action == "toggle_lethargy_demo":
            LETHARGY_DEMO_ACTIVE = not LETHARGY_DEMO_ACTIVE
            if not LETHARGY_DEMO_ACTIVE:
                LETHARGY_DEMO_TARGET_ID = None 
            logger.info(f"Lethargy Demo Mode {'ACTIVATED' if LETHARGY_DEMO_ACTIVE else 'DEACTIVATED'} by {addr}")
            return

        if action == "set_source":
            source = data.get("source")
            if source in ["static_video", "demo_image", "normal_video", "multicam", "multicam_combined"]:
                logger.info(f"Client {addr} requested source change to: {source}")
                CURRENT_SOURCE = source
                if source == "normal_video":
                    if not CAMERA_SOURCES:
                        await initialize_multicam()
                    if CAMERA_SOURCES and CAMERA_SOURCES.get_available_cameras():
                        if len(CAMERA_SOURCES.get_available_cameras()) == 1:
                            CURRENT_CAMERA_INDEX = CAMERA_SOURCES.get_available_cameras()[0]
                        await ws.send_str(json.dumps({
                            "type": "multicam_ready",
                            "cameras": CAMERA_SOURCES.get_available_cameras(),
                            "camera_names": {idx: CAMERA_SOURCES.get_camera_name(idx) for idx in CAMERA_SOURCES.get_available_cameras()}
                        }))
                    else:
                        await check_and_manage_normal_video_task()
                elif source == "static_video":
                    await check_and_manage_static_video_task()
                elif source == "multicam":
                    if not CAMERA_SOURCES:
                        await initialize_multicam()
                    if CAMERA_SOURCES:
                        await ws.send_str(json.dumps({
                            "type": "multicam_available_cameras",
                            "cameras": CAMERA_SOURCES.get_available_cameras(),
                            "camera_names": {idx: CAMERA_SOURCES.get_camera_name(idx) for idx in CAMERA_SOURCES.get_available_cameras()}
                        }))
                elif source == "multicam_combined":
                    if not CAMERA_SOURCES:
                        await initialize_multicam()
                    if CAMERA_SOURCES and CAMERA_SOURCES.get_available_cameras():
                        await ws.send_str(json.dumps({
                            "type": "multicam_combined_ready",
                            "cameras": CAMERA_SOURCES.get_available_cameras()
                        }))
            return

        if action == "switch_camera":
            camera_idx = data.get("camera_index")
            if camera_idx is not None and CAMERA_SOURCES and CAMERA_SOURCES.is_open(camera_idx):
                CURRENT_CAMERA_INDEX = camera_idx
                logger.info(f"Client {addr} switched to camera {camera_idx}")
                await ws.send_str(json.dumps({
                    "type": "camera_switched",
                    "camera_index": camera_idx,
                    "camera_name": CAMERA_SOURCES.get_camera_name(camera_idx)
                }))
            return

        if action == "store_alert":
            alert_data = data.get("alert")
            if alert_data:
                logger.info(f"Client {addr} requested to store a demo alert.")
                alert_data["is_demo"] = True
                asyncio.create_task(add_alert_to_history(alert_data))
            return

        stream = data.get("stream")
        stream_updated = False
        if stream in SUBSCRIPTIONS:
            if action == "subscribe":
                if ws not in SUBSCRIPTIONS[stream]:
                    SUBSCRIPTIONS[stream].add(ws)
                    logger.info(f"Client {addr} subscribed to '{stream}'. Total: {len(SUBSCRIPTIONS[stream])}")
                    stream_updated = True
                    if stream.startswith("multicam_") and stream != "multicam_combined":
                        camera_idx = int(stream.split("_")[1])
                        if CAMERA_SOURCES and CAMERA_SOURCES.is_open(camera_idx):
                            await check_and_manage_multicam_tasks()
                    elif stream == "multicam_combined":
                        await check_and_manage_multicam_combined_task()
            elif action == "unsubscribe":
                if ws in SUBSCRIPTIONS[stream]:
                    SUBSCRIPTIONS[stream].discard(ws)
                    logger.info(f"Client {addr} unsubscribed from '{stream}'. Total: {len(SUBSCRIPTIONS[stream])}")
                    stream_updated = True
        
        if stream_updated:
            if stream == "normal_video": await check_and_manage_normal_video_task()
            elif stream == "static_video": await check_and_manage_static_video_task()
            elif stream == "thermal_video": await check_and_manage_thermal_video_task()
            elif stream.startswith("multicam_") and stream != "multicam_combined":
                await check_and_manage_multicam_tasks()
            elif stream == "multicam_combined":
                await check_and_manage_multicam_combined_task()

    except Exception as e: 
        logger.error(f"Error processing message from {addr}: {e}", exc_info=True)

async def websocket_handler(request):
    addr_str = str(request.remote) if request.remote else "unknown_ws_client"
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info(f"WebSocket connection established from {addr_str}")
    CLIENTS.add(ws)
    ws.addr_str = addr_str

    try:
        history_data = []
        async with history_lock:
            if HISTORY_FILE.is_file() and HISTORY_FILE.stat().st_size > 0:
                try:
                    history_data = json.loads(HISTORY_FILE.read_text())
                except json.JSONDecodeError:
                    logger.error(f"Could not decode history.json, sending empty history.")

        await ws.send_str(json.dumps({"type": "history_load", "data": history_data}))
    except Exception as e:
        logger.error(f"Failed to send history to {addr_str}: {e}")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await handle_websocket_message(ws, addr_str, msg.data)
    finally:
        logger.info(f"WebSocket connection closing for {addr_str}")
        CLIENTS.discard(ws)
        for stream_name in list(SUBSCRIPTIONS.keys()):
            if ws in SUBSCRIPTIONS[stream_name]:
                SUBSCRIPTIONS[stream_name].discard(ws)
                logger.info(f"Client {addr_str} removed from '{stream_name}'.")
                if stream_name == "normal_video": await check_and_manage_normal_video_task()
                elif stream_name == "static_video": await check_and_manage_static_video_task()
                elif stream_name == "thermal_video": await check_and_manage_thermal_video_task()
                elif stream_name.startswith("multicam_") and stream_name != "multicam_combined": await check_and_manage_multicam_tasks()
                elif stream_name == "multicam_combined": await check_and_manage_multicam_combined_task()
    return ws

async def safe_send_binary(ws, data_bytes):
    if ws.closed: return
    try:
        await ws.send_bytes(data_bytes)
    except (ConnectionResetError, asyncio.CancelledError, RuntimeError):
        pass

async def handle_index(request): return web.FileResponse(STATIC_FILES_DIR / 'plan.html')
async def handle_static(request):
    req_path_str = request.match_info.get('filename', '')
    file_path = (STATIC_FILES_DIR / req_path_str).resolve()
    if ".." in req_path_str or not str(file_path).startswith(str(STATIC_FILES_DIR.resolve())):
        return web.Response(status=403, text="Forbidden")
    if file_path.is_file():
        return web.FileResponse(file_path)
    file_path = (STATIC_FILES_DIR / "static" / req_path_str).resolve()
    if file_path.is_file():
        return web.FileResponse(file_path)
    return web.Response(status=404, text="Not Found")

def initialize_demo_image():
    global demo_image
    save_path = STATIC_FILES_DIR / "static" / "demo_image.jpg"
    if save_path.is_file():
        logger.info(f"Loading existing demo image from {save_path}")
        demo_image = cv2.imread(str(save_path))
    else:
        logger.warning(f"Demo image not found. Generating a default image and saving to {save_path}")
        img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.circle(img, (100, 150), 30, (255, 100, 100), -1) 
        cv2.circle(img, (300, 300), 40, (100, 255, 100), -1) 
        cv2.circle(img, (500, 200), 25, (100, 100, 255), -1) 
        demo_image = img
        cv2.imwrite(str(save_path), img)

async def on_app_startup(app_instance):
    global thermal_camera
    logger.info("Application starting up...")
    async with history_lock:
        if not HISTORY_FILE.is_file() or HISTORY_FILE.stat().st_size == 0:
            logger.info(f"History file not found or is empty. Creating/initializing at {HISTORY_FILE}")
            HISTORY_FILE.write_text("[]")
    initialize_demo_image()
    thermal_camera = ThermalCameraManager()
    asyncio.create_task(initialize_yolo_model_async())
    asyncio.create_task(initialize_disease_models_async())
    logger.info("Model initialization processes initiated.")

async def on_app_cleanup(app_instance):
    global MULTICAM_COMBINED_TASK
    logger.info("Application shutting down, cleaning up resources...")
    tasks_to_cancel = [t for t in [normal_video_task, static_video_task, thermal_video_task] if t and not t.done()]
    for task in tasks_to_cancel:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
    if MULTICAM_COMBINED_TASK and not MULTICAM_COMBINED_TASK.done():
        MULTICAM_COMBINED_TASK.cancel()
        try:
            await MULTICAM_COMBINED_TASK
        except asyncio.CancelledError:
            pass
    await release_stream_source(stream_type="normal_video")
    await release_stream_source(stream_type="static_video")
    await release_multicam()
    if thermal_camera: thermal_camera.close()
    logger.info("Cleanup: Shutting down thread pool executor...")
    executor.shutdown(wait=True)
    logger.info("Cleanup: Executor shut down.")

def detect_usb_webcams():
    """Auto-detect available USB webcam indices."""
    detected = []
    import glob
    for device_path in glob.glob('/sys/class/video4linux/video*'):
        idx = int(device_path.split('video')[-1])
        name_file = f"{device_path}/name"
        try:
            with open(name_file, 'r') as f:
                name = f.read().strip()
                if 'Webcam' in name or 'UVC' in name or 'USB' in name:
                    detected.append(idx)
        except:
            pass
    return detected

async def main():
     app = web.Application(logger=logger)
     app.on_startup.append(on_app_startup)
     app.on_cleanup.append(on_app_cleanup)
     app.router.add_get('/ws', websocket_handler)
     app.router.add_get('/', handle_index)
     app.router.add_get('/{filename:.+}', handle_static)
     runner = web.AppRunner(app)
     await runner.setup()
     site = web.TCPSite(runner, HOST, PORT)
     detected_cams = detect_usb_webcams()
     logger.info("-----------------------------------------------------")
     logger.info("Starting PoultryScope Server - RPi Optimized & Fully Featured")
     logger.info(f"Serving on http://{HOST}:{PORT}")
     logger.info(f"CSI Camera via Picamera2: {'Available' if IS_PICAMERA_AVAILABLE else 'Not Available (will use OpenCV fallback)'}")
     logger.info(f"YOLO Detection: {YOLO_MODEL_PATH}")
     logger.info("Disease Classification: ENABLED (ONNX Quantized)")
     logger.info(f"Thermal Camera: {'ENABLED (Hardware/Simulation)' if thermal_camera else 'DISABLED'}")
     logger.info(f"Configured USB Webcams: {WEBCAM_INDICES}")
     logger.info(f"Detected USB Webcams: {detected_cams}")
     logger.info("-----------------------------------------------------")
     await site.start()
     try:
         while True: await asyncio.sleep(3600) 
     except (KeyboardInterrupt, asyncio.CancelledError): pass
     finally:
         logger.info("Shutdown signal received. Cleaning up.")
         await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shutting down.")
    except Exception as e:
        logger.critical(f"Unhandled exception in __main__: {e}", exc_info=True)
