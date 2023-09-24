"""
Step Detection
"""
import numpy as np


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
    if (swing[i - 1] < swing[i]) and np.max(
        stance[i:min(i + window_size, n)]) == stance_threshold:
      steps[i] = True

  return steps, swing, stance
