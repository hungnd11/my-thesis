"""
Sensors Data Processing
"""

import numpy as np
from typing import Tuple


def get_rotation_matrix(gravity: np.ndarray,
                        geomagnetic: np.ndarray,
                        R: np.ndarray | None,
                        g: float = 9.81) -> bool:
  """
  Compute rotation matrix given gravity and geomagnetic vector.
  The implementation was ported from the implementation of Android's SensorManager.

  Args:
    gravity: the gravity (i.e. acceleration) vector
    geomagnetic: the geomagnetic (i.e. magnetic field) vector
    g: the value of gravity, default: 9.81 m/s2
    R: the result rotation matrix, which is a (9, 1) vector or (16, 1) vector.
  
  Returns:
    True if the computation was successful, False otherwise.

  Refs:
    https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/hardware/SensorManager.java#1185
  """
  assert len(gravity) == 3
  assert len(geomagnetic) == 3

  ax = gravity[0]
  ay = gravity[1]
  az = gravity[2]

  norm_sq_a = ax * ax + ay * ay + az * az

  free_fall_gravity_squared = 0.01 * g * g

  if norm_sq_a < free_fall_gravity_squared:
    # gravity is less than 10% of normal value
    return False

  ex = geomagnetic[0]
  ey = geomagnetic[1]
  ez = geomagnetic[2]
  hx = ey * az - ez * ay
  hy = ez * ax - ex * az
  hz = ex * ay - ey * ax

  norm_h = np.sqrt(hx * hx + hy * hy + hz * hz)

  if norm_h < 0.1:
    # device is close to free fall, or close to magnetic north pole, typical values are > 100
    return False

  inv_h = 1.0 / norm_h
  hx *= inv_h
  hy *= inv_h
  hz *= inv_h

  inv_a = 1.0 / np.sqrt(norm_sq_a)
  ax *= inv_a
  ay *= inv_a
  az *= inv_a
  mx = ay * hz - az * hy
  my = az * hx - ax * hz
  mz = ax * hy - ay * hx

  if R is not None:
    if len(R) == 9:
      R[0] = hx
      R[1] = hy
      R[2] = hz
      R[3] = mx
      R[4] = my
      R[5] = mz
      R[6] = ax
      R[7] = ay
      R[8] = az
    elif len(R) == 16:
      R[0] = hx
      R[1] = hy
      R[2] = hz
      R[3] = 0
      R[4] = mx
      R[5] = my
      R[6] = mz
      R[7] = 0
      R[8] = ax
      R[9] = ay
      R[10] = az
      R[11] = 0
      R[12] = 0
      R[13] = 0
      R[14] = 0
      R[15] = 1

  return True


def get_orientation(R: np.ndarray, values: np.ndarray) -> np.ndarray:
  """
  Compute orientation from rotation matrix.
  The implementation was taken from Android's SensorManager implementation.

  Args:
    R: a np.ndarray represents the rotation matrix
    values: a np.ndarray represents the result orientation vector, the orientation vector is in the the order [azimuth, pitch, roll]

  Returns:
    The result orientation vector
  
  Refs:
    https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/hardware/SensorManager.java#1480
  """
  assert len(values) == 3
  assert len(R) == 9 or len(R) == 16

  if len(R) == 9:
    values[0] = np.arctan2(R[1], R[4])
    values[1] = np.arcsin(-R[7])
    values[2] = np.arctan2(-R[6], R[8])
  else:
    values[0] = np.arctan2(R[1], R[5])
    values[1] = np.arcsin(-R[9])
    values[2] = np.arctan2(-R[8], R[10])

  return values


def estimate_acceleration_heading(
    acce: np.ndarray, magn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Estimating acceleration and heading with respect to the earth coordinate system.

  Args:
    acce: a np.ndarray represents the time-series acceleration readings
    magn: a np.ndarray represents the time-series magnetic field readings
  
  Returns:
    a tuple of result heading and acceleration values
  """
  assert len(acce) == len(magn)

  n = len(acce)

  result_heading = np.zeros(n, dtype=np.float32)
  result_acce = np.zeros((n, 3), dtype=np.float32)

  R = np.zeros(9, dtype=np.float64)
  orientation = np.zeros(3, dtype=np.float64)

  for i in range(n):
    result = get_rotation_matrix(acce[i, 1:], magn[i, 1:], R)
    if result:
      _ = get_orientation(R, orientation)
      result_heading[i] = np.degrees(orientation[0])

      inv_R = np.linalg.inv(R.reshape(3, 3))
      result_acce[i, :] = np.matmul(inv_R, acce[i, 1:])
    else:
      result_heading[i] = result_heading[i - 1]
      result_acce[i, :] = result_acce[i - 1, :]

  return result_acce, result_heading


def estimate_heading_from_waypoints(waypoints: np.ndarray) -> np.ndarray:

  gt_x = waypoints[:, 0]
  gt_y = waypoints[:, 1]

  gt_heading_values = np.degrees(
      np.arctan((gt_y[1:] - gt_y[:-1]) / (gt_x[1:] - gt_x[:-1])))

  return gt_heading_values
