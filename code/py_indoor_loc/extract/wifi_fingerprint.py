"""
Utilities for WiFi fingerprint extraction.
"""
import numpy as np
import pandas as pd

from collections import defaultdict
from tqdm import tqdm


def get_band(freq):
  d2400 = np.abs(freq - 2400)
  d5000 = np.abs(freq - 5000)
  band = np.zeros_like(freq)
  band[d2400 < d5000] = 2400
  band[d2400 >= d5000] = 5000
  return band


def extract_bssid_set(wifi_fingerprint_df_list: list[pd.DataFrame],
                      min_times: int = 1000) -> set[str]:
  """
  Extract the set of bssid which occurs at least a specified number of times.

  Args:
    wifi_fingerprint_df_list: a list of 

  Refs:
    https://www.kaggle.com/code/devinanzelmo/wifi-features/notebook
  """
  bssid_sample_count = defaultdict(int)

  for df in wifi_fingerprint_df_list:
    df_bssid_sample_count = df["bssid"].value_counts().to_dict()
    for bssid, count in df_bssid_sample_count.items():
      bssid_sample_count[bssid] += count

  bssid_set = {k for k, v in bssid_sample_count.items() if v >= min_times}

  return bssid_set


def create_fingerprint_vector(group_data: pd.DataFrame,
                              bssid_vector: np.ndarray,
                              not_seen_rssi: float = -1000) -> np.ndarray:
  v = np.zeros_like(bssid_vector, dtype=np.float32) + not_seen_rssi

  for bssid, rssi in group_data[["bssid", "rssi"]].values:
    v[bssid_vector == bssid] = rssi

  return v


def extract_fingerprint_df(fingerprint_files,
                           wifi_band=(2400, 5000),
                           min_samples: int = 0,
                           not_seen_rssi: float = -1000,
                           max_scan_time_gap_ms: float = 2000):
  if isinstance(wifi_band, int):
    wifi_band = {wifi_band}

  wifi_fingerprint_df_list = [pd.read_csv(file) for file in fingerprint_files]

  # Adding frequency band
  for df in wifi_fingerprint_df_list:
    df["freq_band"] = get_band(df["freq"].values)

  wifi_fingerprint_fb_df_list = [
      df[df["freq_band"].isin(wifi_band) &
         (df["sys_ts"] - df["last_seen_ts"] <= max_scan_time_gap_ms)]
      for df in wifi_fingerprint_df_list
  ]

  bssid_set = extract_bssid_set(wifi_fingerprint_fb_df_list,
                                min_times=min_samples)
  print(
      f"The number of BSSIDs with at least {min_samples} samples: {len(bssid_set)}"
  )
  bssid_vector = np.array(list(bssid_set))

  fingerprint_tuples = []
  for df in tqdm(wifi_fingerprint_fb_df_list):
    for (sys_ts, x, y), group_data in df.groupby(["sys_ts", "x", "y"]):
      fingerprint_vector = create_fingerprint_vector(
          group_data, bssid_vector, not_seen_rssi=not_seen_rssi)
      fingerprint_tuples.append((x, y, fingerprint_vector))

  print(f"The number of fingerprints: {len(fingerprint_tuples)}")

  return pd.DataFrame(fingerprint_tuples, columns=["x", "y", "v"]), bssid_vector
