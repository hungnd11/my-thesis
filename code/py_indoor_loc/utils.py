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


class DataListingUtil(object):

  def __init__(self, train_test_assignment: pd.DataFrame,
               experiment_setup: pd.DataFrame, data_dir: str | Path):
    self.train_test_assignment = train_test_assignment
    self.experiment_setup = experiment_setup

    if isinstance(data_dir, str):
      self.data_dir = Path(data_dir)
    else:
      self.data_dir = data_dir

  def list_unlabeled_tracks(self, site_id: str, floor_id: str,
                            p: float) -> list[str]:
    assert p >= 0 and p <= 1

    experiment_setup = self.experiment_setup

    return experiment_setup.loc[(experiment_setup["site_id"] == site_id) &
                                (experiment_setup["floor_id"] == floor_id) &
                                (experiment_setup["dataset"] == "unlabeled") &
                                (experiment_setup["supervision_pct"] == p),
                                "track_id"].values.tolist()

  def list_labeled_tracks(self, site_id: str, floor_id: str,
                          p: float) -> list[str]:
    assert p >= 0 and p <= 1

    experiment_setup = self.experiment_setup

    return experiment_setup.loc[(experiment_setup["site_id"] == site_id) &
                                (experiment_setup["floor_id"] == floor_id) &
                                (experiment_setup["dataset"] == "labeled") &
                                (experiment_setup["floor_id"] == p),
                                "track_id"].values.tolist()

  def list_train_tracks(self, site_id: str, floor_id: str):
    return self._list_track_by_dataset(site_id, floor_id, dataset="train")

  def list_test_track(self, site_id: str, floor_id: str):
    return self._list_track_by_dataset(site_id, floor_id, dataset="test")

  def _list_track_by_dataset(self, site_id: str, floor_id: str, dataset: str):
    train_test_assignment = self.train_test_assignment
    return train_test_assignment.loc[
        (train_test_assignment["site_id"] == site_id) &
        (train_test_assignment["floor_id"] == floor_id) &
        (train_test_assignment["dataset"] == dataset),
        "track_id"].values.tolist()

  def list_train_files(self, site_id: str, floor_id: str):
    return self.list_files(site_id, floor_id,
                           self.list_train_tracks(site_id, floor_id))

  def list_test_files(self, site_id: str, floor_id: str):
    return self.list_files(site_id, floor_id,
                           self.list_train_tracks(site_id, floor_id))

  def list_files(self, site_id: str, floor_id: str, track_list: list[str]):
    return [
        self.data_dir / site_id / floor_id / (track_id + ".txt")
        for track_id in track_list
    ]
