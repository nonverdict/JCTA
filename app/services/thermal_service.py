import time
import numpy as np
import cv2
import logging
from perlin_noise import PerlinNoise
import config

logger = logging.getLogger("PoultryScope.Thermal")


class ThermalService:
    def __init__(self):
        self._noise = PerlinNoise(octaves=2, seed=int(time.time()))
        self._sim_time = 0

    def read_frame(self):
        self._sim_time += 0.05
        # Vectorized Perlin noise generation
        xs = np.arange(32) / 32.0
        ys = np.arange(24) / 24.0
        xv, yv = np.meshgrid(xs, ys)
        # PerlinNoise expects individual coordinate queries; vectorize via numpy vectorize
        vnoise = np.vectorize(lambda x, y: self._noise([x, y, self._sim_time]))
        frame = vnoise(xv, yv)

        # Fixed temperature mapping: 30C to 38C
        frame = 34 + (frame * 4)
        np.clip(frame, 30, 38, out=frame)

        # Fixed color mapping using absolute temperature scale
        # Map 30-38C to 0-255
        norm_frame = ((frame - 30.0) / 8.0 * 255.0).astype(np.uint8)
        color_frame = cv2.applyColorMap(norm_frame, cv2.COLORMAP_INFERNO)
        resized = cv2.resize(color_frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        return resized
