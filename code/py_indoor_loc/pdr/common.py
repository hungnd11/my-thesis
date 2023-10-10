import numpy as np
from py_indoor_loc.sensors import compute_earth_acce_heading, compute_earth_acce_heading_ahrs


def compute_step_heading(step_timestamps: np.ndarray,
                         headings: np.ndarray) -> np.ndarray:
  step_headings = np.zeros((len(step_timestamps), 2))
  step_timestamps_index = 0
  for i in range(0, len(headings)):
    if step_timestamps_index < len(step_timestamps):
      if headings[i, 0] == step_timestamps[step_timestamps_index]:
        step_headings[step_timestamps_index, :] = headings[i, :]
        step_timestamps_index += 1
    else:
      break
  assert step_timestamps_index == len(step_timestamps)

  return step_headings


def compute_rel_positions(stride_lengths: np.ndarray,
                          step_headings: np.ndarray) -> np.ndarray:
  rel_positions = np.zeros((stride_lengths.shape[0], 3))

  rel_positions[:, 0] = stride_lengths[:, 0]
  rel_positions[:, 1] = -stride_lengths[:, 1] * np.sin(step_headings[:, 1])
  rel_positions[:, 2] = stride_lengths[:, 1] * np.cos(step_headings[:, 1])

  return rel_positions


class RelPositionPredictor(object):

  def __init__(self, sd_model, sl_model, name, use_ahrs: bool = True):
    self._sd_model = sd_model
    self._sl_model = sl_model
    self._name = name
    self._use_ahrs = use_ahrs

  @property
  def name(self):
    return self._name

  def predict(self, path_data_collection) -> np.ndarray:

    step_mask = self._sd_model.predict(path_data_collection)
    stride_length = self._sl_model.predict(path_data_collection)

    if self._use_ahrs:
      _, heading = compute_earth_acce_heading_ahrs(path_data_collection.acce,
                                                   path_data_collection.ahrs)
    else:
      _, heading = compute_earth_acce_heading(path_data_collection.acce,
                                              path_data_collection.magn)

    sensor_timestamps = path_data_collection.acce[:, 0]

    heading_with_timestamps = np.vstack((sensor_timestamps, heading)).T
    step_timestamps = sensor_timestamps[step_mask]
    step_headings = compute_step_heading(step_timestamps,
                                         heading_with_timestamps)
    step_headings[:, 1] = np.radians(step_headings[:, 1])

    stride_lengths = np.zeros((len(step_timestamps), 2), dtype=np.float64)
    stride_lengths[:, 0] = step_timestamps
    if self._sl_model.returns_at_step():
      stride_lengths[:, 1] = stride_length
    else:
      stride_lengths[:, 1] = stride_length[step_mask]

    rel_positions = compute_rel_positions(stride_lengths, step_headings)

    return rel_positions


def compute_cumulative_step_positions(
    rel_step_positions: np.ndarray) -> np.ndarray:
  assert len(rel_step_positions) > 0
  assert len(rel_step_positions[0]) == 3

  results = np.zeros_like(rel_step_positions, dtype=np.float64)

  results[:, 0] = rel_step_positions[:, 0]
  results[:, 1] = np.cumsum(rel_step_positions[:, 1])
  results[:, 2] = np.cumsum(rel_step_positions[:, 2])

  return results