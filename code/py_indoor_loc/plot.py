"""
Ploting Utilities
"""
import pandas as pd
import matplotlib.pyplot as plt

from shapely import Polygon
from typing import Any


def plot_n_unique_bssids_by_rssi(wifi_location_df: pd.DataFrame,
                                 label: str = "num_unique_bssids",
                                 ax: plt.Axes | None = None,
                                 plot_kwargs: dict | None = None):
  if ax is None:
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

  if plot_kwargs is None:
    plot_kwargs = dict()

  (wifi_location_df.groupby("rssi").agg({
      "bssid": pd.Series.nunique
  }).rename(columns={
      "bssid": label
  }).plot(marker="o", alpha=0.7, label=label, ax=ax, **plot_kwargs))

  ax.set_xlabel("RSS (dBm)")
  ax.set_ylabel("Number of unique BSSIDs")
  ax.set_title("The number of unique BSSIDs by RSS level")
  ax.set_xticks(range(-100, 0, 5))
  ax.grid()

  return ax


def plot_floor_map(floor_polygons: list[Polygon],
                   store_polygons: list[Polygon],
                   ax: plt.Axes | None = None,
                   floor_plot_kwargs: dict[str, Any] | None = None,
                   store_plot_kwargs: dict[str, Any] | None = None):

  if floor_plot_kwargs is None:
    floor_plot_kwargs = dict()

  if store_plot_kwargs is None:
    store_plot_kwargs = dict()

  if ax is None:
    _, ax = plt.subplots(1, 1, figsize=(12, 6))

  for floor_polygon in floor_polygons:
    x, y = floor_polygon.exterior.xy
    ax.plot(x, y, color="black", **floor_plot_kwargs)

  for store_polygon in store_polygons:
    x, y = store_polygon.exterior.xy
    ax.plot(x, y, color="grey", **store_plot_kwargs)

  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title("Floor map")

  return ax