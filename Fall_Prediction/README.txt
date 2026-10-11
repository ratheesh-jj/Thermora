BITS-2 prepared IMU dataset for THERMORA fall-detection baseline

INPUT CHANNELS
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

WINDOWING
20 Hz motion sampling rate
1.0-second windows = 20 samples
0.5-second stride = 10 samples

LABELS
0 = ADL / non-fall
1 = fall

FILES
bits2_imu_windows.npz
  X: shape (N, 20, 6)
  y: shape (N,)
window_metadata.csv
file_statistics.csv
summary.json

IMPORTANT
This is a first training-ready benchmark. BITS-2 recordings have recording-level
labels, so all windows from a fall recording are currently labeled as fall.
For deployment-quality fall-event detection, we should later refine the positive
windows around the actual fall event if event timing can be established.

The timestamp column is not used as a feature because the extracted BITS-2 CSV
timestamps are distributed in rounded scientific notation. Motion sampling is
documented by the dataset authors as 20 Hz.

Participant split:
train = users 1-29
validation = users 30-35
test = users 36-41
