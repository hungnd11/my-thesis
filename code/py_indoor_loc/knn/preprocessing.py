"""
Preprocessing for kNN
"""
import numpy as np
import pandas as pd

from typing import Iterable
from py_indoor_loc.extract.wifi_fingerprint import extract_fingerprint_df


def translate(X: np.ndarray,
              X_bssid: np.ndarray,
              target_bssid: np.ndarray,
              default_rss: float = -100):

  X_target = np.ones((X.shape[0], target_bssid.shape[0])) * default_rss

  for i, t_bssid in enumerate(target_bssid):
    indices = np.where(X_bssid == t_bssid)[0]
    if len(indices) == 1:
      X_target[:, i] = X[:, indices[0]]

  return X_target


def combine_bssid(bssid_vector_list: Iterable[Iterable[str]]) -> np.ndarray:
  bssid_set = set()
  for bssid_vector in bssid_vector_list:
    bssid_set.update(bssid_vector)
  return np.array(list(bssid_set))


def min_filter(X: np.ndarray, min_rss: float = -100) -> np.ndarray:
  X_t = np.zeros_like(X)
  loc = np.where(X >= min_rss)
  X_t[loc] = X[loc] - min_rss
  return X_t


def extract_train_test(
    train_fingerprint_df: pd.DataFrame,
    train_bssid: np.ndarray,
    test_fingerprint_df: pd.DataFrame,
    test_bssid: np.ndarray,
    extract_fingerprint_kwargs: dict | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

  if extract_fingerprint_kwargs is None:
    extract_fingerprint_kwargs = dict()

  X_train = np.vstack(train_fingerprint_df["v"].values.tolist())
  y_train = train_fingerprint_df[["x", "y"]].values

  X_test = np.vstack(test_fingerprint_df["v"].values.tolist())
  y_test = test_fingerprint_df[["x", "y"]].values

  bssid_vector = combine_bssid([train_bssid, test_bssid])

  X_train_translated = translate(X_train, train_bssid, bssid_vector, -100)
  X_test_translated = translate(X_test, test_bssid, bssid_vector, -100)
  assert X_train_translated.shape[1] == X_test_translated.shape[1]
  assert X_train.shape[0] == X_train_translated.shape[0]
  assert X_test.shape[0] == X_test_translated.shape[0]

  return X_train_translated, y_train, X_test_translated, y_test, bssid_vector
