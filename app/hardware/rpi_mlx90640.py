import adafruit_mlx90640
import board
import busio
import numpy as np


class MLX90640Reader:
    """MLX90640 thermal IR array sensor reader for Raspberry Pi via I2C.

    Requires:
        - Adafruit-Blinka (busio, board)
        - adafruit-circuitpython-mlx90640
        - I2C enabled on Raspberry Pi (GPIO 2 = SDA, GPIO 3 = SCL)
    """

    def __init__(self, refresh_rate=adafruit_mlx90640.RefreshRate.REFRESH_2_HZ):
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
            self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
            self.mlx.refresh_rate = refresh_rate
            self.frame = np.zeros((24 * 32,), dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MLX90640: {e}")

    def get_frame(self):
        """Read a fresh 24x32 temperature frame in Celsius.

        Returns:
            np.ndarray: (24, 32) array of temperatures, or None on failure.
        """
        try:
            self.mlx.getFrame(self.frame)
            # MLX90640 can produce NaN on bad pixels; replace with nearest valid
            if np.isnan(self.frame).any():
                # Simple fallback: interpolate NaNs with mean of valid pixels
                mean_val = np.nanmean(self.frame)
                np.nan_to_num(self.frame, copy=False, nan=mean_val)
            return self.frame.reshape((24, 32))
        except ValueError as e:
            # getFrame often raises ValueError on I2C timing issues; safe to retry
            return None
        except Exception as e:
            raise RuntimeError(f"MLX90640 read failed: {e}")

    def read_temperature(self):
        """Legacy helper: returns (avg_c, avg_f) or (None, None).

        Note: get_frame() is preferred for new code.
        """
        frame = self.get_frame()
        if frame is not None:
            avg_c = float(np.mean(frame))
            avg_f = (avg_c * 9.0 / 5.0) + 32.0
            return avg_c, avg_f
        return None, None

    def close(self):
        try:
            self.i2c.deinit()
        except Exception:
            pass
