"""
Step Detection
"""
import abc
import logging
import numpy as np

from py_indoor_loc.model import PathDataCollection
from py_indoor_loc.lang import override
from py_indoor_loc.sensors import compute_earth_acce_heading

from scipy.ndimage import median_filter


class AbcSDModel(abc.ABC):

  def __init__(self) -> None:
    self._logger = logging.getLogger(type(self).__name__)

  def predict(self, path_data_collection: PathDataCollection) -> np.ndarray:
    raise NotImplementedError()


class LocalAccVarianceSDModel(AbcSDModel):
  """
  Local Acceleration Variance Step Detector Model
  """

  def __init__(self,
               window_size: int = 8,
               swing_threshold: float = 2.,
               stance_threshold: float = 1.):
    self._window_size = window_size
    self._swing_threshold = swing_threshold
    self._stance_threshold = stance_threshold

  def compute_mean(self, acce: np.ndarray) -> np.ndarray:
    return compute_mean(acce, self._window_size)

  @override
  def predict(self, path_data_collection: PathDataCollection) -> np.ndarray:
    earth_acce, _ = compute_earth_acce_heading(path_data_collection.acce,
                                               path_data_collection.magn)
    acce_magnitude = np.linalg.norm(earth_acce, axis=1)
    acce_var = compute_local_acc_variance(acce_magnitude,
                                          window_size=self._window_size)
    step_mask, _, _ = compute_step_positions(
        acce_var,
        swing_threshold=self._swing_threshold,
        stance_threshold=self._stance_threshold,
        window_size=self._window_size)

    return step_mask


class AngularRateSDModel(AbcSDModel):

  def __init__(self,
               median_filter_window_size: int = 8,
               stance_threshold: float = 1.) -> None:
    super().__init__()
    self._median_filter_window_size = median_filter_window_size
    self._stance_threshold = stance_threshold

  @override
  def predict(self, path_data_collection: PathDataCollection) -> np.ndarray:
    total_angular_velocity = np.linalg.norm(path_data_collection.gyro[:, 1:],
                                            axis=1)
    f_total_angular_velocity = median_filter(
        total_angular_velocity,
        size=self._median_filter_window_size).astype(np.float64)

    n = len(f_total_angular_velocity)
    step_signal = np.ones(n, dtype=np.uint8)
    step_signal[f_total_angular_velocity <= self._stance_threshold] = 0

    step_indices = np.where(step_signal[1:] > step_signal[:-1])[0]
    step_mask = np.array([False] * n)
    step_mask[step_indices + 1] = True

    return step_mask


def compute_mean(acce: np.ndarray, window_size: int = 15) -> np.ndarray:
  n = len(acce)
  w_mean = np.zeros(n, dtype=np.float32)
  for i in range(n):
    start_idx = max(0, i - window_size)
    end_idx = min(n, i + window_size + 1)
    w_mean[i] = np.mean(acce[start_idx:end_idx])
  return w_mean


def compute_local_acc_variance(acce: np.ndarray, window_size: int = 15):
  """
  Compute the local acceleration variance to highlight the foot activity and to remove gravity.
  """
  n = len(acce)
  variance = np.zeros_like(acce, dtype=np.float32)
  for i in range(n):
    start_idx = max(0, i - window_size)
    end_idx = min(n, i + window_size + 1)
    variance[i] = np.var(acce[start_idx:end_idx])

  return np.sqrt(variance)


def compute_step_positions(acce_var: np.ndarray,
                           swing_threshold: float = 2,
                           stance_threshold: float = 1,
                           window_size: int = 15):
  n = len(acce_var)
  steps = np.array([False] * n)

  # swing[i] = T1 if var > T1, 0 otherwise
  swing = np.zeros(n, dtype=np.float32)
  swing[acce_var > swing_threshold] = swing_threshold

  # stance[i] = T2 if var < T2, 0 otherwise
  stance = np.zeros(n, dtype=np.float32)
  stance[acce_var < stance_threshold] = stance_threshold

  for i in range(1, n):
    if (swing[i - 1] > swing[i]) and np.max(
        stance[i:min(i + window_size, n)]) == stance_threshold:
      steps[i] = True

  return steps, swing, stance
