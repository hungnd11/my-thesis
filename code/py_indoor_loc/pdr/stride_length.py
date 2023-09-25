"""
Stride Length Estimation
"""

import abc
import logging

import numpy as np

from py_indoor_loc.model import PathDataCollection
from py_indoor_loc.lang import override
from py_indoor_loc.sensors import compute_earth_acce_heading
from py_indoor_loc.pdr.step_detection import AbcSDModel
from scipy import signal


class AbcSLModel(abc.ABC):

  def __init__(self) -> None:
    self._logger = logging.getLogger(type(self).__name__)

  def predict(self, path_data_collection: PathDataCollection) -> np.ndarray:
    raise NotImplementedError()


class WeinbergSLModel(AbcSLModel):
  """
  Weinberg SL Model.
  """

  def __init__(self,
               cutoff_frequency=3,
               filter_order=4,
               fs=50,
               K: float = 0.364,
               window_size: int = 15) -> None:
    super().__init__()
    self._lp_params = {
        "cutoff_frequency": cutoff_frequency,
        "filter_order": filter_order,
        "fs": fs,
    }
    self._K = K
    self._window_size = window_size

  @override
  def predict(self, path_data_collection: PathDataCollection) -> np.ndarray:
    earth_acce, earth_heading = compute_earth_acce_heading(
        path_data_collection.acce, path_data_collection.magn)
    acce_magnitude = np.linalg.norm(earth_acce, axis=1)
    n = len(acce_magnitude)
    f_acce_magnitude = filter_lp(acce_magnitude, **self._lp_params)

    sl = np.zeros(n, dtype=np.float32)

    for k in range(0, n):
      j_min = max(0, k - self._window_size)
      j_max = min(n - 1, k + self._window_size)
      sl[k] = self._K * np.power(
          f_acce_magnitude[j_min:j_max + 1].max() -
          f_acce_magnitude[j_min:j_max + 1].min(), 0.25)

    return sl


class ZUPTSLModel(AbcSLModel):

  def __init__(self,
               sd_model: AbcSDModel,
               fs: int = 50,
               window_size: int = 8) -> None:
    super().__init__()
    self._sd_model = sd_model
    self._fs = fs
    self._window_size = window_size

  @override
  def predict(self, path_data_collection: PathDataCollection) -> np.ndarray:
    # TODO: Using quatenion
    acce, _ = compute_earth_acce_heading(path_data_collection.acce,
                                         path_data_collection.magn)
    vx = integrate(acce[:, 0], fs=self._fs)
    vy = integrate(acce[:, 1], fs=self._fs)
    vz = integrate(acce[:, 2], fs=self._fs)

    cvx = np.zeros_like(vx)
    cvy = np.zeros_like(vy)
    cvz = np.zeros_like(vz)

    step_mask = self._sd_model.predict(path_data_collection)
    step_indices = np.where(step_mask)[0]
    n = len(acce)

    prev_ik = 0
    prev_mu = (0, 0, 0)

    for ik in step_indices:

      min_ik, max_ik = max(0,
                           ik - self._window_size), min(n,
                                                        ik + self._window_size)
      mu_k_x = np.mean(vx[min_ik:max_ik + 1])
      mu_k_y = np.mean(vy[min_ik:max_ik + 1])
      mu_k_z = np.mean(vz[min_ik:max_ik + 1])

      n_samples = ik - prev_ik
      pmk_x, pmk_y, pmk_z = prev_mu

      for i in range(prev_ik, ik):
        cvx[i] = vx[i] - (pmk_x * (ik - i) + mu_k_x * (i - prev_ik)) / n_samples
        cvy[i] = vy[i] - (pmk_y * (ik - i) + mu_k_y * (i - prev_ik)) / n_samples
        cvz[i] = vz[i] - (pmk_z * (ik - i) + mu_k_z * (i - prev_ik)) / n_samples
      prev_ik = ik
      prev_mu = (mu_k_x, mu_k_y, mu_k_z)

    prev_ik = 0
    delta_pk = []

    for ik in step_indices:
      delta_pk_x = cvx[prev_ik:ik].sum() / self._fs
      delta_pk_y = cvy[prev_ik:ik].sum() / self._fs
      delta_pk_z = cvz[prev_ik:ik].sum() / self._fs

      delta_pk.append((delta_pk_x, delta_pk_y, delta_pk_z))

    sl = np.sqrt([pkn * pkn + pke * pke for (pkn, pke, _) in delta_pk])

    return sl


def filter_lp(acce_magnitude: np.ndarray,
              cutoff_frequency: int = 3,
              filter_order: int = 4,
              fs: int = 50) -> np.ndarray:

  b, a = signal.butter(filter_order,
                       cutoff_frequency,
                       'low',
                       analog=False,
                       fs=fs)
  # Apply the filter to the noisy signal
  f_acce_magnitude = signal.filtfilt(b, a, acce_magnitude)

  return f_acce_magnitude


def integrate(a: np.ndarray, fs: int = 50) -> np.ndarray:
  n = len(a)
  prev_v = 0
  result = np.zeros(n, dtype=np.float32)
  for i in range(n):
    result[i] = prev_v + a[i] / fs
    prev_v = result[i]
  return result