"""
Utilities
"""

import numpy as np
import pandas as pd

from typing import Tuple
from pathlib import Path


def rotate(xs: np.ndarray, ys: np.ndarray,
           a: float) -> Tuple[np.ndarray, np.ndarray]:
  """
  Rotate points by an angle.

  Args:
    xs: The x-coordinate of points
    ys: The y-coordinate of points
    a: the angle, in degrees.
  
  Returns:
    a tuple of x-coordinates and y-coordinates of points after rotation

  Refs:
    https://en.wikipedia.org/wiki/Rotation_of_axes_in_two_dimensions
  """
  xs = np.atleast_1d(xs)
  ys = np.atleast_1d(ys)

  r_xs = xs * np.cos(a) + ys * np.sin(a)
  r_ys = -1 * xs * np.sin(a) + ys * np.cos(a)

  return r_xs, r_ys


def inverse_rotate(xs: np.ndarray, ys: np.ndarray,
                   a: float) -> Tuple[np.ndarray, np.ndarray]:
  ir_xs = xs * np.cos(a) - ys * np.sin(a)
  ir_ys = xs * np.sin(a) + ys * np.cos(a)
  return ir_xs, ir_ys