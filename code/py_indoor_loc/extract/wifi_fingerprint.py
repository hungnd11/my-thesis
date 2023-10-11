"""
Utilities for WiFi fingerprint extraction.
"""
import numpy as np
import pandas as pd

from typing import Iterable
from collections import defaultdict
from tqdm import tqdm


def get_band(freq: Iterable[int]) -> np.ndarray:

  if not isinstance(freq, np.ndarray):
    freq = np.array(freq)

  diff_2400 = np.abs(freq - 2400)
  diff_5000 = np.abs(freq - 5000)
  band = np.zeros_like(freq)
  band[diff_2400 < diff_5000] = 2400
  band[diff_2400 >= diff_5000] = 5000

  return band


def extract_bssid_set(wifi_location_df_list: list[pd.DataFrame],
                      min_app_times: int = -1) -> set[str]:
  """
  Extract the set of bssid which occurs at least a specified number of times.

  Args:
    wifi_location_df_list: a list of wifi location dataframes
    min_app_times: the minimum number of appearances across all dataframes
  
  Returns:
    A set of BSSIDs with at least min_app_time appearances
  
  Refs:
    https://www.kaggle.com/code/devinanzelmo/wifi-features/notebook
  """
  bssid_sample_count = defaultdict(int)

  for df in wifi_location_df_list:
    df_bssid_sample_count = df["bssid"].value_counts().to_dict()
    for bssid, count in df_bssid_sample_count.items():
      bssid_sample_count[bssid] += count

  bssid_set = {k for k, v in bssid_sample_count.items() if v >= min_app_times}

  return bssid_set


def create_fingerprint_vector(group_data: pd.DataFrame,
                              bssid_vector: np.ndarray,
                              ap_not_seen_rss: float = -1000) -> np.ndarray:
  v = np.zeros_like(bssid_vector, dtype=np.float32) + ap_not_seen_rss

  for bssid, rssi in group_data[["bssid", "rssi"]].values.tolist():
    v[bssid_vector == bssid] = rssi

  return v


def read_wifi_location_df(file_path: str) -> pd.DataFrame:
  wifi_freq_to_channel_map = {
      2412: 1,
      2417: 2,
      2422: 3,
      2427: 4,
      2432: 5,
      2437: 6,
      2442: 7,
      2447: 8,
      2452: 9,
      2457: 10,
      2462: 11,
      2467: 12,
      2472: 13,
      2482: 14,
      5180: 36,
      5200: 40,
      5220: 44,
      5240: 48,
      5260: 52,
      5280: 56,
      5300: 60,
      5745: 149,
      5765: 153,
      5785: 157,
      5805: 161,
      5825: 165,
  }
  wifi_location_df = pd.read_csv(file_path)

  wifi_location_df["rssi"] = wifi_location_df["rssi"].astype(np.float32)
  wifi_location_df["freq"] = wifi_location_df["freq"].astype(np.int32)
  wifi_location_df["freq_band"] = get_band(wifi_location_df["freq"].values)
  wifi_location_df["channel"] = wifi_location_df["freq"].map(
      wifi_freq_to_channel_map)

  return wifi_location_df


def read_wifi_location_df_list(file_path_list: list[str]) -> list[pd.DataFrame]:
  frames = []
  for file_path in file_path_list:
    try:
      frames.append(read_wifi_location_df(file_path))
    except Exception as e:
      print(e)
  return frames


def extract_fingerprint_df(wifi_location_df_list: list[pd.DataFrame],
                           wifi_band=(2400, 5000),
                           min_app_times: int = 0,
                           not_seen_rssi: float = -1000,
                           max_scan_time_gap_ms: float = 2000,
                           tqdm_enabled: bool = False,
                           verbose: bool = False):

  for i, df in enumerate(wifi_location_df_list):
    try:
      verify_wifi_location_df(df)
    except ValueError as ve:
      raise ValueError(
          f"Failed to verify wifi_location_df_list[{i}], caused by: {str(ve)}")

  if isinstance(wifi_band, int):
    wifi_band = {wifi_band}

  fb_wifi_location_df_list = [
      df.loc[df["freq_band"].isin(wifi_band) &
             (df["sys_ts"] - df["last_seen_ts"] <= max_scan_time_gap_ms)]
      for df in wifi_location_df_list
  ]

  bssid_set = extract_bssid_set(fb_wifi_location_df_list,
                                min_app_times=min_app_times)
  if verbose:
    print(
        f"The number of BSSIDs with at least {min_app_times} samples: {len(bssid_set)}"
    )
  bssid_vector = np.array(list(bssid_set))

  fingerprint_tuples = []
  iterator = tqdm(
      fb_wifi_location_df_list) if tqdm_enabled else fb_wifi_location_df_list

  for df in iterator:
    for (sys_ts, x, y), group_data in df.groupby(["sys_ts", "x", "y"]):
      fingerprint_vector = create_fingerprint_vector(
          group_data, bssid_vector, ap_not_seen_rss=not_seen_rssi)
      fingerprint_tuples.append((sys_ts, x, y, fingerprint_vector))

  if verbose:
    print(f"The number of fingerprints: {len(fingerprint_tuples)}")

  return pd.DataFrame(fingerprint_tuples, columns=["sys_ts", "x", "y",
                                                   "v"]), bssid_vector


def verify_wifi_location_df(df: pd.DataFrame) -> None:
  data_columns = set(df.columns)
  for col in ["sys_ts", "last_seen_ts", "freq_band", "bssid", "rssi", "x", "y"]:
    if col not in data_columns:
      raise ValueError("Missing column: " + col)
