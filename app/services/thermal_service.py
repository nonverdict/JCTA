import time
import numpy as np
import cv2
import logging
from perlin_noise import PerlinNoise
import config

logger = logging.getLogger("PoultryScope.Thermal")


class ThermalService:
    def __init__(self):
        self._real_reader = None
        self._real_frame = None
        self._noise = PerlinNoise(octaves=2, seed=int(time.time()))
        self._sim_time = 0
        self.source = "simulated"
        self._last_avg_temp = 34.0

        # Attempt to initialize real MLX90640 sensor
        try:
            from app.hardware.rpi_mlx90640 import MLX90640Reader
            self._real_reader = MLX90640Reader()
            self.source = "real"
            logger.info("ThermalService: Real MLX90640 sensor initialized.")
        except Exception as e:
            logger.warning(f"ThermalService: MLX90640 not available ({e}). Using simulation.")
            self._real_reader = None
            self.source = "simulated"

    def _read_simulated(self):
        """Generate simulated 24x32 thermal frame with Perlin noise."""
        self._sim_time += 0.05
        xs = np.arange(32) / 32.0
        ys = np.arange(24) / 24.0
        xv, yv = np.meshgrid(xs, ys)
        vnoise = np.vectorize(lambda x, y: self._noise([x, y, self._sim_time]))
        frame = vnoise(xv, yv)
        frame = 34 + (frame * 4)
        np.clip(frame, 30, 38, out=frame)
        return frame

    def _read_real(self):
        """Read real MLX90640 sensor frame."""
        try:
            frame = self._real_reader.get_frame()
            if frame is not None:
                return frame
            logger.debug("MLX90640 read returned None, falling back to last frame.")
        except Exception as e:
            logger.warning(f"MLX90640 read error: {e}")
        # Return last known good frame or simulated if none exists
        if self._real_frame is not None:
            return self._real_frame
        return self._read_simulated()

    def read_frame(self):
        """Return a colored thermal image (320x240) suitable for streaming."""
        if self._real_reader is not None:
            self._real_frame = self._read_real()
            temp_frame = self._real_frame
        else:
            temp_frame = self._read_simulated()

        # Compute avg temperature from the frame (nanmean for robustness with bad pixels)
        self._last_avg_temp = float(np.nanmean(temp_frame))

        # Normalize to 0-255 for colormap (use 20C to 45C range for real sensor flexibility)
        norm_frame = ((temp_frame - 20.0) / 25.0 * 255.0)
        np.clip(norm_frame, 0, 255, out=norm_frame)
        norm_frame = norm_frame.astype(np.uint8)
        color_frame = cv2.applyColorMap(norm_frame, cv2.COLORMAP_INFERNO)
        resized = cv2.resize(color_frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        return resized

    def get_avg_temperature(self):
        return self._last_avg_temp

    def get_temperature_frame(self):
        """Return raw temperature data (24x32) for analysis."""
        if self._real_reader is not None and self._real_frame is not None:
            return self._real_frame
        return self._read_simulated()
