import numpy as np


def detect_turn_points(step_headings_f: np.ndarray, 
                       turn_threshold_degrees: float = 30) -> np.ndarray:
  
  turn_threshold = np.radians(turn_threshold_degrees)

  turn_mask = np.array([False] * len(step_headings_f))

  last_turn_step_index = 0
  for current_step_index in range(len(step_headings_f)):
    if current_step_index == 0:
      continue

    prev_mean = np.mean(step_headings_f[last_turn_step_index:current_step_index])
    if np.abs(step_headings_f[current_step_index] - prev_mean) >= turn_threshold:
      turn_mask[current_step_index] = True
      last_turn_step_index = current_step_index

  return turn_mask
