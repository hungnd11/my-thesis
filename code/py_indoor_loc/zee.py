"""
Implement Zee Paper
"""

import numpy as np

STATE_IDLE = "IDLE"
STATE_WALKING = "WALKING"


def norm_acf(a: np.ndarray, m: int, t: int) -> float:
  mu_m, sigma_m = np.mean(a[m:m + t], axis=0), np.std(a[m:m + t], axis=0)
  mu_mt, sigma_mt = np.mean(a[m + t:m + t * 2],
                            axis=0), np.std(a[m + t:m + t * 2], axis=0)

  n_acf = (np.sum((a[m:m + t] - mu_m) * (a[m + t:m + t * 2] - mu_mt), axis=0) /
           t / sigma_m / sigma_mt)

  return np.mean(n_acf)


def max_norm_acf(a: np.ndarray, m: int, t_min: int = 40, t_max: int = 100):
  """
  Finding the maximum normalized auto-correlation.
  """
  t_best = t_min
  n_acf_best = norm_acf(a, m, t_best)
  for t in range(t_min, t_max + 1):
    n_acf = norm_acf(a, m, t)
    if n_acf > n_acf_best:
      t_best = t
      n_acf_best = n_acf
  return n_acf_best, t_best


def get_state(acce: np.ndarray,
              m: int,
              prev_state: str | None = None,
              prev_t_opt: int | None = None):

  if prev_t_opt is None:
    t_min, t_max = 40, 100

  t_min, t_max = max(40, prev_t_opt - 10), min(100, prev_t_opt + 10)

  n_acf, t_opt = max_norm_acf(acce, m=m, t_min=t_min, t_max=t_max)

  sigma = np.mean(np.std(acce[m:m + t_opt], axis=0))

  if sigma < 0.01:
    return STATE_IDLE, t_opt

  if n_acf > 0.7:
    return STATE_WALKING, t_opt

  return prev_state, t_opt
