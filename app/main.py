import asyncio
import concurrent.futures
from pathlib import Path
from aiohttp import web
from app.services.detection_service import DetectionService, CameraManager
from app.services.analytics_service import AnalyticsService
from app.services.model_service import ModelService
from app.routes.camera_routes import handle_index, handle_static
from app.routes.api_routes import websocket_handler, get_history
from app.routes.status_routes import get_status
from app.services.thermal_service import ThermalService
from app.utils.logger import logger
import config
import cv2
import numpy as np
import json


def _load_or_create_demo_image(demo_path):
    """Load demo image. If it's the old placeholder (mostly black) or missing,
    try to extract a frame from chicken_demo.mp4, otherwise generate a clean placeholder."""
    img = None
    if demo_path.is_file():
        img = cv2.imread(str(demo_path))
        if img is not None and img.mean() > 15:
            logger.info(f"Loaded demo image from {demo_path}")
            return img
        elif img is not None:
            logger.warning("Existing demo image is the old placeholder (mostly black). Regenerating...")
        else:
            logger.warning(f"Failed to read demo image at {demo_path}")

    # Try to extract a frame from the demo video
    video_path = Path(config.STATIC_VIDEO_PATH)
    if video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            success, frame = cap.read()
            cap.release()
            if success and frame is not None:
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                cv2.imwrite(str(demo_path), frame)
                logger.info(f"Extracted demo frame from {video_path}")
                return frame

    # Generate a clean placeholder with text
    placeholder = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), np.uint8)
    # Dark slate background
    placeholder[:, :] = (30, 35, 40)
    # Subtle border
    cv2.rectangle(placeholder, (20, 20), (620, 460), (60, 70, 80), 2)
    # Text lines
    text_lines = [
        ("Januya Demo Mode", 0.6, (200, 200, 200), 80),
        ("Replace static/assets/demo_image.jpg", 0.45, (150, 155, 160), 200),
        ("with your own image to customize", 0.45, (150, 155, 160), 240),
    ]
    for text, scale, color, y in text_lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = (config.FRAME_WIDTH - tw) // 2
        cv2.putText(placeholder, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    cv2.imwrite(str(demo_path), placeholder)
    logger.info(f"Generated clean demo placeholder at {demo_path}")
    return placeholder


async def on_startup(app):
    app["executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    app["history_lock"] = asyncio.Lock()
    app["history_file"] = config.HISTORY_FILE
    app["clients"] = set()
    app["active_streams"] = set()
    app["subscriptions"] = {
        "normal_video": set(),
        "static_video": set(),
        "demo_image": set(),
        "thermal_video": set(),
        "debug_stream": set(),
    }
    app["lethargy_demo_active"] = False
    app["current_source"] = "normal_video"

    # Ensure history file exists
    hist_file = Path(config.HISTORY_FILE)
    hist_file.parent.mkdir(parents=True, exist_ok=True)
    if not hist_file.is_file() or hist_file.stat().st_size == 0:
        hist_file.write_text("[]")
        logger.info(f"Initialized empty history file at {hist_file}")

    demo_path = config.STATIC_DIR / "assets" / "demo_image.jpg"
    app["demo_image_data"] = _load_or_create_demo_image(demo_path)

    app["detector"] = DetectionService(app)
    app["models"] = ModelService()
    app["manager"] = AnalyticsService(app)
    app["thermal"] = ThermalService()

    # Static video source
    static_path = Path(config.STATIC_VIDEO_PATH)
    if static_path.is_file():
        app["static_source"] = CameraManager(config.STATIC_VIDEO_PATH, is_file=True)
        app["static_source"].start()
        logger.info(f"Static video source ready: {static_path}")
    else:
        logger.warning(f"Static video not found at {static_path}. Static stream will be unavailable.")
        app["static_source"] = None

    # Normal webcam source
    app["normal_source"] = CameraManager(config.WEBCAM_INDICES[0])
    if not app["normal_source"].start():
        logger.error("Failed to start Live Camera on primary index. Checking backup index...")
        app["normal_source"].release()
        app["normal_source"] = CameraManager(config.WEBCAM_INDICES[1])
        if not app["normal_source"].start():
            logger.error("Failed to start Live Camera on backup index as well. Live stream will be unavailable.")
        else:
            logger.info(f"Live camera started on backup index {config.WEBCAM_INDICES[1]}")
    else:
        logger.info(f"Live camera started on primary index {config.WEBCAM_INDICES[0]}")

    await app["detector"].initialize()
    await app["models"].initialize(app["executor"])

    logger.info("Startup complete. All services initialized.")


async def on_cleanup(app):
    logger.info("Shutting down application...")

    # Stop all active broadcast loops by clearing active_streams
    app["active_streams"].clear()
    # Give loops a moment to see the change
    await asyncio.sleep(0.2)

    # Release camera sources
    if app.get("static_source"):
        app["static_source"].release()
        logger.info("Static source released.")
    if app.get("normal_source"):
        app["normal_source"].release()
        logger.info("Normal source released.")

    # Shutdown executor
    executor = app.get("executor")
    if executor:
        executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down.")

    # Close websocket connections gracefully
    for ws in list(app.get("clients", set())):
        if not ws.closed:
            await ws.close()
    logger.info("Cleanup complete.")


def create_app():
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # Specific Routes first
    app.router.add_get("/", handle_index)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/api/history", get_history)
    app.router.add_get("/api/status", get_status)

    # Catch-all static route LAST
    app.router.add_get("/{filename:.+}", handle_static)

    return app
