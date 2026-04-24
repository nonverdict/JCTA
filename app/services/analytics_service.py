import json
import asyncio
import cv2
import time
import numpy as np
from app.utils.logger import logger
import config


class AnalyticsService:
    def __init__(self, app):
        self.app = app
        self._static_retry_delay = 0.0
        self._task_lock = asyncio.Lock()

    async def add_alert_to_history(self, alert_data):
        async with self.app["history_lock"]:
            file = self.app["history_file"]
            try:
                file.parent.mkdir(parents=True, exist_ok=True)
                history = json.loads(file.read_text()) if file.is_file() else []
                history.insert(0, alert_data)
                if len(history) > config.MAX_HISTORY_ENTRIES:
                    history = history[:config.MAX_HISTORY_ENTRIES]
                    logger.info(f"History truncated to {config.MAX_HISTORY_ENTRIES} entries")
                file.write_text(json.dumps(history, indent=4))
            except (IOError, OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to write history: {e}")
                return

        # Broadcast wrapped alert so frontend can distinguish it from debug updates
        payload = json.dumps({"type": "alert", "data": alert_data})
        send_tasks = []
        for ws in list(self.app["clients"]):
            if not ws.closed:
                try:
                    send_tasks.append(ws.send_str(payload))
                except Exception:
                    pass
        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)

    async def _wrapped_broadcast_loop(self, stream_type):
        """Wraps broadcast_loop to catch exceptions and prevent silent task death."""
        try:
            await self.broadcast_loop(stream_type)
        except Exception:
            logger.exception(f"Broadcast loop for {stream_type} crashed")
        finally:
            self.app["active_streams"].discard(stream_type)
            logger.info(f"Broadcast task for {stream_type} ended")

    async def _wrapped_thermal_loop(self):
        """Wraps thermal_loop to catch exceptions and prevent silent task death."""
        try:
            await self.thermal_loop()
        except Exception:
            logger.exception("Thermal loop crashed")
        finally:
            self.app["active_streams"].discard("thermal_video")
            logger.info("Thermal broadcast task ended")

    async def broadcast_loop(self, stream_type):
        loop = asyncio.get_running_loop()
        logger.info(f"Starting broadcast for {stream_type}")

        while stream_type in self.app["active_streams"]:
            start = time.monotonic()

            # Determine frame source based on stream type
            if stream_type == "demo_image":
                frame = self.app.get("demo_image_data")
                if frame is not None:
                    frame = frame.copy()
            elif stream_type == "static_video":
                source = self.app.get("static_source")
                if source is None or not source.is_open():
                    await asyncio.sleep(0.5)
                    continue
                frame = await loop.run_in_executor(self.app["executor"], source.read)
                if frame is None:
                    self._static_retry_delay = min(self._static_retry_delay + 0.5, 5.0)
                    logger.warning(f"Static video read failed, retrying in {self._static_retry_delay}s")
                    await asyncio.sleep(self._static_retry_delay)
                    source.start()
                    continue
                else:
                    self._static_retry_delay = 0.0
            else:  # normal_video
                source = self.app.get("normal_source")
                if source is None or not source.is_open():
                    await asyncio.sleep(0.5)
                    continue
                frame = await loop.run_in_executor(self.app["executor"], source.read)

            if frame is None:
                await asyncio.sleep(0.1)
                continue

            res = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            idx = self.app["detector"].next_frame_index()

            proc_frame, tracks = await loop.run_in_executor(
                self.app["executor"],
                self.app["detector"].process_frame,
                frame,
                res,
                idx,
            )

            try:
                alerts, flock_avg = await loop.run_in_executor(
                    self.app["executor"],
                    self.app["detector"].check_anomalies,
                    idx,
                )
                for alert in alerts:
                    await self.add_alert_to_history(alert)
            except Exception as e:
                logger.error(f"Anomaly checking failed: {e}")
                alerts, flock_avg = [], 0.0

            if self.app["subscriptions"].get("debug_stream"):
                try:
                    debug_payload = json.dumps({
                        "type": "debug_update",
                        "data": {
                            "flock_avg_movement": flock_avg,
                            "tracked_objects": self.app["detector"].get_debug_state(),
                        },
                    })
                    for ws in list(self.app["subscriptions"]["debug_stream"]):
                        if not ws.closed:
                            try:
                                await ws.send_str(debug_payload)
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Debug stream send failed: {e}")

            ret, buffer = cv2.imencode(
                ".jpg", proc_frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]
            )
            if not ret:
                logger.warning("JPEG encoding failed")
                await asyncio.sleep(0.1)
                continue

            msg = bytes([1]) + buffer.tobytes()
            subscribers = list(self.app["subscriptions"].get(stream_type, set()))
            send_tasks = []
            for ws in subscribers:
                if not ws.closed:
                    try:
                        send_tasks.append(ws.send_bytes(msg))
                    except Exception:
                        pass
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)

            elapsed = time.monotonic() - start
            sleep_time = max(0, config.FRAME_DELAY - elapsed)
            if sleep_time == 0 and elapsed > config.FRAME_DELAY * 2:
                logger.debug(f"Slow frame processing: {elapsed:.3f}s")
            await asyncio.sleep(sleep_time)

        self.app["active_streams"].discard(stream_type)
        logger.info(f"Stopped broadcast for {stream_type}")

    async def thermal_loop(self):
        stream_type = "thermal_video"
        logger.info("Starting thermal broadcast")

        while stream_type in self.app["active_streams"]:
            start = time.monotonic()

            try:
                frame = self.app["thermal"].read_frame()
            except Exception as e:
                logger.error(f"Thermal frame read failed: {e}")
                await asyncio.sleep(0.5)
                continue

            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                await asyncio.sleep(0.1)
                continue

            msg = bytes([2]) + buffer.tobytes()
            subscribers = list(self.app["subscriptions"].get(stream_type, set()))
            send_tasks = []
            for ws in subscribers:
                if not ws.closed:
                    try:
                        send_tasks.append(ws.send_bytes(msg))
                    except Exception:
                        pass
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)

            await asyncio.sleep(
                max(0, config.THERMAL_FRAME_DELAY - (time.monotonic() - start))
            )

        self.app["active_streams"].discard(stream_type)
        logger.info("Stopped thermal broadcast")

    async def check_and_manage_tasks(self):
        async with self._task_lock:
            for stream in ["normal_video", "static_video", "demo_image"]:
                sub_count = len(self.app["subscriptions"].get(stream, set()))
                if sub_count > 0 and stream not in self.app["active_streams"]:
                    self.app["active_streams"].add(stream)
                    asyncio.create_task(self._wrapped_broadcast_loop(stream))
                elif sub_count == 0 and stream in self.app["active_streams"]:
                    self.app["active_streams"].discard(stream)

            thermal_subs = len(self.app["subscriptions"].get("thermal_video", set()))
            if thermal_subs > 0 and "thermal_video" not in self.app["active_streams"]:
                self.app["active_streams"].add("thermal_video")
                asyncio.create_task(self._wrapped_thermal_loop())
            elif thermal_subs == 0 and "thermal_video" in self.app["active_streams"]:
                self.app["active_streams"].discard("thermal_video")

    async def handle_message(self, ws, addr, msg_str):
        try:
            data = json.loads(msg_str)
            action = data.get("action")

            if action == "subscribe":
                stream = data.get("stream")
                if stream in self.app["subscriptions"]:
                    self.app["subscriptions"][stream].add(ws)
                    await self.check_and_manage_tasks()

            elif action == "unsubscribe":
                stream = data.get("stream")
                if stream in self.app["subscriptions"]:
                    self.app["subscriptions"][stream].discard(ws)

            elif action == "set_source":
                source = data.get("source")
                self.app["current_source"] = source
                logger.info(f"Client {addr} set source to {source}")
                await self.check_and_manage_tasks()

            elif action == "toggle_lethargy_demo":
                self.app["lethargy_demo_active"] = not self.app.get(
                    "lethargy_demo_active", False
                )
                logger.info(
                    f"Lethargy demo toggled: {self.app['lethargy_demo_active']}"
                )

            elif action == "store_alert":
                alert = data.get("alert")
                if alert and isinstance(alert, dict):
                    msg = alert.get("message", "")
                    if not isinstance(msg, str) or len(msg) > 500:
                        logger.warning(f"Invalid alert message from {addr}")
                        return
                    alert_type = alert.get("type", "")
                    if alert_type not in ("disease_alert", "behavior_alert"):
                        logger.warning(f"Invalid alert type from {addr}: {alert_type}")
                        return
                    alert["is_demo"] = True
                    await self.add_alert_to_history(alert)

            elif action == "inject_disease_demo":
                disease = data.get("disease")
                if disease:
                    alert = self.app["detector"].inject_disease(disease)
                    if alert:
                        logger.info(f"[DEMO] Injected disease: {disease}")
                        await self.add_alert_to_history(alert)
                    else:
                        logger.warning("[DEMO] No tracks available for disease injection")

        except Exception as e:
            logger.error(f"Error handling WS message: {e}")
