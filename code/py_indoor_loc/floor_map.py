import json
import numpy as np

from shapely.geometry.polygon import Polygon


def read_floor_data(floor_dir_path: str):
  if not floor_dir_path.endswith("/"):
    floor_dir_path += "/"

  floor_info_filename = floor_dir_path + "floor_info.json"
  floor_map_filename = floor_dir_path + "geojson_map.json"

  with open(floor_info_filename) as f:
    floor_info = json.load(f)

  with open(floor_map_filename) as f:
    floor_map = json.load(f)

  return floor_info, floor_map


def parse_polygon_coordinates(geojson_feature: dict):
  polygons = []

  feature_type = geojson_feature["geometry"]["type"]

  if feature_type == "MultiPolygon":
    for coord_set in geojson_feature["geometry"]["coordinates"]:
      polygons.append(np.array(coord_set[0], dtype=np.float64))

  elif feature_type == "Polygon":
    polygons.append(
        np.array(geojson_feature["geometry"]["coordinates"][0],
                 dtype=np.float64))

  return polygons


def find_bound(floor_coordinates):
  assert len(floor_coordinates) > 0

  x_min, y_min = np.min(floor_coordinates[0], axis=0)
  x_max, y_max = np.max(floor_coordinates[0], axis=0)

  for coords in floor_coordinates[1:]:
    _x_min, _y_min = np.min(coords, axis=0)
    _x_max, _y_max = np.max(coords, axis=0)
    x_min = np.min([x_min, _x_min])
    y_min = np.min([y_min, _y_min])
    x_max = np.max([x_max, _x_max])
    y_max = np.max([y_max, _y_max])

  return x_min, y_min, x_max, y_max


def scale(coords,
          x_min,
          y_min,
          x_max,
          y_max,
          width_meter,
          height_meter,
          inplace=False):
  if not inplace:
    coords = coords.copy()

  coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min) * width_meter
  coords[:, 1] = (coords[:, 1] - y_min) / (y_max - y_min) * height_meter

  return coords


def extract_floor_map_geometries(floor_map, floor_info, transform=None):
  if transform is None:
    transform = lambda t: t

  # Finding the bound of the floormap
  floor_coordinates = parse_polygon_coordinates(floor_map["features"][0])
  x_min, y_min, x_max, y_max = find_bound(floor_coordinates)

  width_meter = floor_info["map_info"]["width"]
  height_meter = floor_info["map_info"]["height"]

  # Scaling floormap
  for coords in floor_coordinates:
    scale(coords,
          x_min,
          y_min,
          x_max,
          y_max,
          width_meter,
          height_meter,
          inplace=True)
    coords[:] = transform(coords)

  # Extract store polygons
  store_coordinates = []

  for geojson_feature in floor_map["features"][1:]:
    polygons = parse_polygon_coordinates(geojson_feature)
    # In-place normalization
    for coords in polygons:
      scale(
          coords,
          x_min,
          y_min,
          x_max,
          y_max,
          width_meter,
          height_meter,
          inplace=True,
      )
      coords[:] = transform(coords)
      store_coordinates.append(coords)

  # Scaling transformed coordinates
  x_min, y_min, x_max, y_max = find_bound(floor_coordinates)

  for coords in floor_coordinates:
    scale(coords,
          x_min,
          y_min,
          x_max,
          y_max,
          width_meter,
          height_meter,
          inplace=True)

  for coords in store_coordinates:
    scale(coords,
          x_min,
          y_min,
          x_max,
          y_max,
          width_meter,
          height_meter,
          inplace=True)

  floor_polygons = [Polygon(coords) for coords in floor_coordinates]
  store_polygons = [Polygon(coords) for coords in store_coordinates]

  return (
      floor_polygons,
      store_polygons,
      x_min,
      y_min,
      x_max,
      y_max,
      width_meter,
      height_meter,
  )


def transform_rotation(rotation_angle):

  def impl(coords):
    r_xs, r_ys = rotate(coords[:, 0], coords[:, 1], rotation_angle)
    return np.vstack((r_xs, r_ys)).T

  return impl


def rotate(xs, ys, a):
  xs = np.atleast_1d(xs)
  ys = np.atleast_1d(ys)

  r_xs = xs * np.cos(a) + ys * np.sin(a)
  r_ys = -1 * xs * np.sin(a) + ys * np.cos(a)

  return r_xs, r_ys
