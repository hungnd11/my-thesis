import abc

import numpy as np

from py_indoor_loc.lang import override


class AbcTurnPointDetector(abc.ABC):
  @abc.abstractmethod
  def __call__(self, step_headings: np.ndarray) -> np.ndarray:
    raise NotImplementedError()
  

class TurnPointDetector(AbcTurnPointDetector):
  def __init__(self, turn_threshold_degrees: float = 30) -> None:
    super().__init__()
    self._turn_threshold = np.radians(turn_threshold_degrees)
  
  @override
  def __call__(self, step_headings: np.ndarray) -> np.ndarray:
    turn_mask = np.array([False] * len(step_headings))

    last_turn_step_index = 0
    for current_step_index in range(len(step_headings)):
      if current_step_index == 0:
        continue

      prev_mean = np.mean(step_headings[last_turn_step_index:current_step_index])
      if np.abs(step_headings[current_step_index] - prev_mean) >= self._turn_threshold:
        turn_mask[current_step_index] = True
        last_turn_step_index = current_step_index

    return turn_mask
