"""
Zee Implementation
"""

import numpy as np
import pandas as pd
import scipy
import shapely.vectorized

from dataclasses import dataclass, field
from shapely import Polygon
from shapely import ops as shapely_ops
from sklearn.neighbors import NearestNeighbors

STATE_IDLE = "IDLE"
STATE_WALKING = "WALKING"


def norm_acf(a: np.ndarray, m: int, t: int) -> float:
  mu_m, sigma_m = np.mean(a[m:m + t], axis=0), np.std(a[m:m + t], axis=0)
  mu_mt, sigma_mt = np.mean(a[m + t:m + t * 2],
                            axis=0), np.std(a[m + t:m + t * 2], axis=0)

  n_acf = (np.sum((a[m:m + t] - mu_m) * (a[m + t:m + t * 2] - mu_mt), axis=0) /
           t / sigma_m / sigma_mt)

  return np.mean(n_acf)


def max_norm_acf(a: np.ndarray,
                 m: int,
                 t_min: int = 40,
                 t_max: int = 100) -> tuple[float, int]:
  """
  Finding the maximum normalized auto-correlation.
  """
  t_best = t_min
  n_acf_best = norm_acf(a, m, t_best)
  for t in range(t_min, t_max + 1):
    if (m + t >= len(a)) or (m + 2 * t >= len(a)):
      break
    n_acf = norm_acf(a, m, t)
    if n_acf > n_acf_best:
      t_best = t
      n_acf_best = n_acf
  return n_acf_best, t_best


def get_state(acce: np.ndarray,
              m: int,
              prev_state: str | None = None,
              prev_t_opt: int | None = None) -> tuple[str | None, int]:

  if prev_t_opt is None:
    t_min, t_max = 40, 100
  else:
    t_min, t_max = max(40, prev_t_opt - 10), min(100, prev_t_opt + 10)

  n_acf, t_opt = max_norm_acf(acce, m=m, t_min=t_min, t_max=t_max)

  sigma = np.mean(np.std(acce[m:m + t_opt], axis=0))

  if sigma < 0.01:
    return STATE_IDLE, t_opt

  if n_acf > 0.7:
    return STATE_WALKING, t_opt

  return prev_state, t_opt


def estimate_heading_offset(acce: np.ndarray, fs: int) -> float:
  """
  Estimate heading offset.
  """
  parallel = np.linalg.norm(acce[:, :2], axis=1)
  f_parallel, amp_parallel = scipy.signal.periodogram(parallel, fs=fs)
  idx = np.argmax(amp_parallel)
  fx = -1 * acce[idx * 2, 0]
  fy = acce[idx * 2, 1]

  return np.degrees(np.arctan(fx / fy) % (2 * np.pi))


def extract_hallway(floor_polygons: list[Polygon],
                    store_polygons: list[Polygon],
                    buffer_distance: float = 1.0):
  union_floor_polygon = shapely_ops.unary_union(floor_polygons).buffer(
      buffer_distance)
  union_store_polygon = shapely_ops.unary_union(store_polygons)
  return union_floor_polygon.difference(union_store_polygon)


@dataclass
class SearchGrid(object):
  grid_points: np.ndarray
  # The resolution (in meters) of the grid
  grid_resolution: float

  nn: NearestNeighbors = field(init=False)

  def __post_init__(self):
    self.nn = NearestNeighbors(n_neighbors=5,
                               radius=2 * self.grid_resolution,
                               algorithm="ball_tree").fit(self.grid_points)

  @property
  def n_points(self):
    return self.grid_points.shape[0]


def create_search_grid(floor_polygons,
                       store_polygons,
                       width_meter,
                       height_meter,
                       grid_resolution,
                       tol: float = 1.0) -> SearchGrid:
  union_floor_polygon = shapely_ops.unary_union(floor_polygons).buffer(tol)
  union_store_polygon = shapely_ops.unary_union(store_polygons).buffer(-tol)
  floor_search_space = union_floor_polygon.difference(union_store_polygon)

  grid_xs = np.arange(0, width_meter, grid_resolution) + grid_resolution / 2
  grid_ys = np.arange(0, height_meter, grid_resolution) + grid_resolution / 2

  grid_x, grid_y = np.meshgrid(grid_xs, grid_ys)

  search_space_mask = shapely.vectorized.contains(floor_search_space, grid_x,
                                                  grid_y)
  ss_x, ss_y = grid_x[search_space_mask], grid_y[search_space_mask]
  grid_points = np.vstack((ss_x, ss_y)).T

  return SearchGrid(grid_points, grid_resolution)


@dataclass
class APFStep(object):
  # particles being considered in this step
  particles: np.ndarray
  # sample_index is the index of the sample in the IMU data sequence
  sample_index: int
  # parent[i] = j iff the j-the particle in the previous step is the parent of the current particle
  parent: np.ndarray
  # keep_mask[i] = True if the i-th particle is kept, i.e. it has at least 1 child in the next step
  keep_mask: np.ndarray = field(init=False)

  def __post_init__(self):
    self.keep_mask = np.array([False] * len(self.particles))


def init_apf(
    search_grid: SearchGrid,
    placement_offset_center: float,
    placement_offset_loc: float = 10.,
    n_particles: int = 10000,
    stride_length_range: tuple[float, float] = (0.5, 1.2),
    initial_location_center: tuple[float, float] | None = None,
    initial_location_radius: float = 10.0,
) -> np.ndarray:

  # If we know the distribution of the initial location
  if isinstance(initial_location_center, tuple):
    initial_location_center = np.array(initial_location_center)
    initial_dist = np.linalg.norm(search_grid.grid_points -
                                  initial_location_center,
                                  axis=1)
    eligible_grid_points = search_grid.grid_points[initial_dist <
                                                   initial_location_radius]
  else:
    eligible_grid_points = search_grid.grid_points

  n_search_grid_points = eligible_grid_points.shape[0]

  # Range for stride length
  stride_length_min, stride_length_max = stride_length_range

  particles = np.zeros((n_particles, 4), dtype=np.float64)

  # Initial random locations
  ss_idx = np.random.randint(n_search_grid_points, size=n_particles)
  particles[:, 0] = eligible_grid_points[ss_idx, 0]
  particles[:, 1] = eligible_grid_points[ss_idx, 1]

  # Initial random stride length
  particles[:, 2] = np.random.uniform(stride_length_min,
                                      stride_length_max,
                                      size=n_particles)

  # Initial placement offset
  fwd_placement_offsets = np.random.normal(placement_offset_center,
                                           placement_offset_loc,
                                           size=n_particles // 2)
  inv_placement_offsets = np.random.normal(placement_offset_center + 180,
                                           placement_offset_loc,
                                           size=n_particles // 2)
  particles[:, 3] = np.hstack((fwd_placement_offsets, inv_placement_offsets))
  np.random.shuffle(particles[:, 3])

  return particles


def run_apf(acce: np.ndarray,
            heading: np.ndarray,
            init_particles: np.ndarray,
            search_grid: SearchGrid,
            turn_delta_heading: float = 20,
            magnetic_offset_error_scale: float = 5,
            stride_length_variation: float = 0.1,
            verbose: bool = True) -> list[APFStep]:

  n_particles = len(init_particles)

  acce_magnitude = np.linalg.norm(acce, axis=1)

  step_idx = 0
  it = 0
  prev_state = None
  prev_t_opt = None
  particle_indices = np.arange(n_particles)

  # eliminated = true if the particle was eliminated in the next step, false otherwise
  parent = np.ones(n_particles, dtype=np.int32) * -1
  history = [APFStep(particles=init_particles, parent=parent, sample_index=0)]

  while True:
    it += 1

    if (prev_t_opt is not None) and (step_idx + 2 * prev_t_opt
                                     >= len(acce_magnitude) - 1):
      break

    state, t_opt = get_state(acce_magnitude,
                             m=step_idx,
                             prev_state=prev_state,
                             prev_t_opt=prev_t_opt)

    step_freq = t_opt // 2
    next_step_idx = step_idx + step_freq

    if verbose:
      print(
          f"Iteration {it}: state = {state}, step_freq = {step_freq} samples/step, sample_idx = {step_idx} -> {next_step_idx}"
      )

    if state == "WALKING":
      last_apf_step = history[-1]

      last_particles = last_apf_step.particles.copy()
      particles = last_particles.copy()
      parent = np.ones(n_particles, dtype=np.int32) * -1

      # Update
      magnetic_offset = np.random.normal(
          loc=0,
          scale=np.radians(magnetic_offset_error_scale),
          size=n_particles)
      stride_length = particles[:, 2] + np.random.uniform(
          -stride_length_variation, stride_length_variation,
          size=n_particles) * particles[:, 2]

      # Predict with fractional update if possible
      turn_points = []
      for i in range(step_idx + 1, next_step_idx):
        if np.abs(heading[i] - heading[i - 1]) > turn_delta_heading:
          turn_points.append(i)

      if len(turn_points) == 0:
        # No fractional update
        sensor_headings = np.radians(heading[step_idx:next_step_idx].mean())
        angle = particles[:, 3] + sensor_headings + magnetic_offset
        particles[:, 0] = particles[:, 0] - stride_length * np.sin(angle)
        particles[:, 1] = particles[:, 1] + stride_length * np.cos(angle)
      else:
        # Fractional update
        turn_points = [step_idx, *turn_points, next_step_idx]
        for i, j in zip(turn_points[:-1], turn_points[1:]):
          sensor_headings = np.radians(heading[i:j].mean())
          angle = particles[:, 3] + sensor_headings + magnetic_offset
          particles[:, 0] = particles[:, 0] - stride_length * (
              j - i) / step_freq * np.sin(angle)
          particles[:, 1] = particles[:, 1] + stride_length * (
              j - i) / step_freq * np.cos(angle)

      # Resampling
      # 1/ In order to replace each eliminated particle, a new particle is randomly chosen from the particle set at the previous step AND updated.
      # 2/ To enable traceback, each particle after the k-th step maintains a link to its parent particle
      dists, indices = search_grid.nn.kneighbors(particles[:, :2],
                                                 n_neighbors=1)
      keep = dists[:, 0] < search_grid.grid_resolution
      n_eliminated = np.count_nonzero(~keep)

      parent[keep] = particle_indices[keep]

      if n_eliminated > 0:
        # TODO: Try to fill these things for at least some times
        parent[~keep] = np.random.choice(particle_indices, size=n_eliminated)
        particles[~keep] = last_particles[parent[~keep]]

        magnetic_offset = np.random.normal(
            loc=0,
            scale=np.radians(magnetic_offset_error_scale),
            size=n_eliminated)
        stride_length = particles[~keep, 2] * (
            1 + np.random.uniform(-stride_length_variation,
                                  stride_length_variation,
                                  size=n_eliminated))

        # TODO: This fractional update is too vulnerable to noise, should be skipped
        turn_points = []
        for i in range(step_idx + 1, next_step_idx):
          if np.abs(heading[i] - heading[i - 1]) > turn_delta_heading:
            turn_points.append(i)

        if len(turn_points) == 0:
          # No fractional update
          sensor_headings = np.radians(heading[step_idx:next_step_idx].mean())
          angle = particles[~keep, 3] + sensor_headings + magnetic_offset
          particles[~keep,
                    0] = particles[~keep, 0] + stride_length * np.cos(angle)
          particles[~keep,
                    1] = particles[~keep, 1] + stride_length * np.sin(angle)
        else:
          # Fractional update
          turn_points = [step_idx, *turn_points, next_step_idx]
          for i, j in zip(turn_points[:-1], turn_points[1:]):
            sensor_headings = np.radians(heading[i:j].mean())
            angle = particles[~keep, 3] + sensor_headings + magnetic_offset
            particles[~keep, 0] = particles[
                ~keep, 0] + stride_length * (j - i) / step_freq * np.cos(angle)
            particles[~keep, 1] = particles[
                ~keep, 1] + stride_length * (j - i) / step_freq * np.sin(angle)

        # This method is time consuming
        dists, indices = search_grid.nn.kneighbors(particles[:, :2],
                                                   n_neighbors=1)
        keep = dists[:, 0] < search_grid.grid_resolution
        n_eliminated = np.count_nonzero(~keep)

      # Append history
      history.append(
          APFStep(particles=particles,
                  parent=parent,
                  sample_index=next_step_idx))

    prev_state = state
    prev_t_opt = t_opt
    step_idx = next_step_idx

  # Backward belief propagation

  # All particles in the last step will be keep
  last_idx = len(history) - 1
  history[last_idx].keep_mask[:] = True
  last_parent = history[last_idx].parent
  last_keep_mask = history[last_idx].keep_mask

  while last_idx > 0:
    # Move to the previous step
    last_idx -= 1

    # In this step, the i-th particle is keep iff it has a children and its children is keep
    history[last_idx].keep_mask[last_parent[last_keep_mask]] = True

    # Update the last parent
    last_parent = history[last_idx].parent
    last_keep_mask = history[last_idx].keep_mask

  return history


def create_track_waypoints(sensor_ts: np.ndarray,
                           history: list[APFStep]) -> pd.DataFrame:
  track_waypoints = []

  for step in history:
    step_point = step.particles[step.keep_mask, :2].mean(axis=0).tolist()
    step_time = sensor_ts[step.sample_index]
    track_waypoints.append((step_time, *step_point))

  return pd.DataFrame(track_waypoints, columns=["sys_ts", "x", "y"])
