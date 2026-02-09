import adafruit_mlx90640
import board
import busio
import numpy as np

class MLX90640Reader:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        self.frame = np.zeros((24 * 32,))

    def read_temperature(self):
        try:
            self.mlx.getFrame(self.frame)
            average_temp_c = np.mean(self.frame)
            average_temp_f = (average_temp_c * 9.0 / 5.0) + 32.0
            return average_temp_c, average_temp_f
        except ValueError as e:
            print(f"Failed to read temperature, retrying. Error: {str(e)}")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return None, None
        
    def close(self):
        self.i2c.deinit()
        print("I2C bus closed.")

"""
# Example usage:
if __name__ == "__main__":
    reader = MLX90640Reader()
    try:
        while True:
            temp_c, temp_f = reader.read_temperature()
            if temp_c is not None:
                print(f"Average Temperature: {temp_c:.2f} °C / {temp_f:.2f} °F")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        reader.close()
"""