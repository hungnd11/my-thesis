"""
Step Detection
"""
import abc
import logging
import numpy as np

from py_indoor_loc.model import PathDataCollection
from py_indoor_loc.lang import override
from py_indoor_loc.sensors import compute_earth_acce_heading, compute_earth_acce_heading_ahrs

from scipy.ndimage import median_filter


class AbcSDModel(abc.ABC):

  def __init__(self, use_ahrs: bool = True) -> None:
    self._logger = logging.getLogger(type(self).__name__)

  def predict(self, path_data_collection: PathDataCollection) -> np.ndarray:
    raise NotImplementedError()


class AcceBasedSDModel(AbcSDModel):

  def __init__(self, use_ahrs: bool = True) -> None:
    super().__init__()
    self._use_ahrs = use_ahrs

  def compute_earth_acce_heading(
      self, path_data_collection: PathDataCollection
  ) -> tuple[np.ndarray, np.ndarray]:
    if self._use_ahrs:
      return compute_earth_acce_heading_ahrs(path_data_collection.acce,
                                             path_data_collection.ahrs)
    else:
      return compute_earth_acce_heading(path_data_collection.acce,
                                        path_data_collection.magn)


class LocalAcceVarianceSDModel(AcceBasedSDModel):
  """
  Local Acceleration Variance Step Detector Model
  """

  def __init__(self,
               window_size: int = 8,
               swing_threshold: float = 2.,
               stance_threshold: float = 1.,
               use_ahrs: bool = True):
    super().__init__(use_ahrs=use_ahrs)
    self._window_size = window_size
    self._swing_threshold = swing_threshold
    self._stance_threshold = stance_threshold

  def compute_mean(self, acce: np.ndarray) -> np.ndarray:
    return compute_mean(acce, self._window_size)

  @override
  def predict(self, path_data_collection: PathDataCollection) -> np.ndarray:
    earth_acce, _ = self.compute_earth_acce_heading(path_data_collection)
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


class AcfSDModel(AcceBasedSDModel):

  STATE_IDLE = "IDLE"
  STATE_WALKING = "WALKING"

  def __init__(self,
               t_min: int = 40,
               t_max: int = 100,
               use_ahrs: bool = True) -> None:
    super().__init__(use_ahrs=use_ahrs)
    assert t_max >= t_min, "t_max ({}) < t_min ({})".format(t_max, t_min)

    self._t_min = t_min
    self._t_max = t_max

  @override
  def predict(self, path_data_collection: PathDataCollection) -> np.ndarray:
    earth_acce, earth_heading = self.compute_earth_acce_heading(
        path_data_collection)
    acce_magnitude = np.linalg.norm(earth_acce, axis=1)

    step_indices = [0]
    n = len(acce_magnitude)
    step_idx = 0
    prev_state, prev_t_opt = None, None
    while True:
      if (prev_t_opt is not None) and (step_idx + 2 * prev_t_opt >= n - 1):
        break
      state, t_opt = AcfSDModel.get_state(acce_magnitude,
                                          m=step_idx,
                                          prev_state=prev_state,
                                          prev_t_opt=prev_t_opt)
      step_freq = t_opt // 2
      next_step_idx = step_idx + step_freq

      if state == AcfSDModel.STATE_WALKING:
        step_indices.append(next_step_idx)

      prev_state = state
      prev_t_opt = t_opt
      step_idx = next_step_idx

    step_mask = np.array([False] * n)
    step_mask[step_indices] = True
    return step_mask

  @classmethod
  def norm_acf(cls, a: np.ndarray, m: int, t: int) -> float:
    mu_m, sigma_m = np.mean(a[m:m + t], axis=0), np.std(a[m:m + t], axis=0)
    mu_mt, sigma_mt = np.mean(a[m + t:m + t * 2],
                              axis=0), np.std(a[m + t:m + t * 2], axis=0)

    n_acf = (np.sum(
        (a[m:m + t] - mu_m) * (a[m + t:m + t * 2] - mu_mt), axis=0) / t /
             sigma_m / sigma_mt)

    return np.mean(n_acf)

  @classmethod
  def max_norm_acf(cls,
                   a: np.ndarray,
                   m: int,
                   t_min: int = 40,
                   t_max: int = 100) -> tuple[float, int]:
    """
    Finding the maximum normalized auto-correlation.
    """
    t_best = t_min
    n_acf_best = cls.norm_acf(a, m, t_best)
    for t in range(t_min, t_max + 1):
      if (m + t >= len(a)) or (m + 2 * t >= len(a)):
        break
      n_acf = cls.norm_acf(a, m, t)
      if n_acf > n_acf_best:
        t_best = t
        n_acf_best = n_acf
    return n_acf_best, t_best

  @classmethod
  def get_state(cls,
                acce: np.ndarray,
                m: int,
                prev_state: str | None = None,
                prev_t_opt: int | None = None,
                max_idle_sigma: float = 0.01,
                min_n_acf: float = 0.7) -> tuple[str | None, int]:

    if prev_t_opt is None:
      t_min, t_max = 40, 100
    else:
      t_min, t_max = max(40, prev_t_opt - 10), min(100, prev_t_opt + 10)

    n_acf, t_opt = cls.max_norm_acf(acce, m=m, t_min=t_min, t_max=t_max)

    sigma = np.mean(np.std(acce[m:m + t_opt], axis=0))

    if sigma < max_idle_sigma:
      return cls.STATE_IDLE, t_opt

    if n_acf > min_n_acf:
      return cls.STATE_WALKING, t_opt

    return prev_state, t_opt


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
