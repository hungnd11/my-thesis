from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PathDataCollection:
  acce: np.ndarray
  acce_uncali: np.ndarray
  gyro: np.ndarray
  gyro_uncali: np.ndarray
  magn: np.ndarray
  magn_uncali: np.ndarray
  ahrs: np.ndarray
  wifi: np.ndarray
  ibeacon: np.ndarray
  waypoint: np.ndarray


def parse_data_text(data_lines: list[str]) -> PathDataCollection:
  acce = []
  acce_uncali = []
  gyro = []
  gyro_uncali = []
  magn = []
  magn_uncali = []
  ahrs = []
  wifi = []
  ibeacon = []
  waypoint = []

  for line_data in data_lines:
    try:
      line_data = line_data.strip()
      if not line_data or line_data[0] == '#':
        continue

      line_data = line_data.split('\t')

      if line_data[1] == 'TYPE_ACCELEROMETER':
        acce.append([
            int(line_data[0]),
            float(line_data[2]),
            float(line_data[3]),
            float(line_data[4])
        ])
        continue

      if line_data[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
        acce_uncali.append([
            int(line_data[0]),
            float(line_data[2]),
            float(line_data[3]),
            float(line_data[4])
        ])
        continue

      if line_data[1] == 'TYPE_GYROSCOPE':
        gyro.append([
            int(line_data[0]),
            float(line_data[2]),
            float(line_data[3]),
            float(line_data[4])
        ])
        continue

      if line_data[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
        gyro_uncali.append([
            int(line_data[0]),
            float(line_data[2]),
            float(line_data[3]),
            float(line_data[4])
        ])
        continue

      if line_data[1] == 'TYPE_MAGNETIC_FIELD':
        magn.append([
            int(line_data[0]),
            float(line_data[2]),
            float(line_data[3]),
            float(line_data[4])
        ])
        continue

      if line_data[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
        magn_uncali.append([
            int(line_data[0]),
            float(line_data[2]),
            float(line_data[3]),
            float(line_data[4])
        ])
        continue

      if line_data[1] == 'TYPE_ROTATION_VECTOR':
        ahrs.append([
            int(line_data[0]),
            float(line_data[2]),
            float(line_data[3]),
            float(line_data[4])
        ])
        continue

      if line_data[1] == 'TYPE_WIFI':
        sys_ts = line_data[0]
        ssid = line_data[2]
        bssid = line_data[3]
        rssi = line_data[4]
        freq = line_data[5]
        lastseen_ts = line_data[6]
        wifi_data = [sys_ts, ssid, bssid, rssi, freq, lastseen_ts]
        wifi.append(wifi_data)
        continue

      if line_data[1] == 'TYPE_BEACON':
        ts = line_data[0]
        uuid = line_data[2]
        major = line_data[3]
        minor = line_data[4]
        rssi = line_data[6]
        ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
        ibeacon.append(ibeacon_data)
        continue

      if line_data[1] == 'TYPE_WAYPOINT':
        waypoint.append(
            [int(line_data[0]),
             float(line_data[2]),
             float(line_data[3])])

    except Exception as e:
      print(f"Failed to processing line: {line_data}. Caused by {str(e)}")

  acce = np.array(acce)
  acce_uncali = np.array(acce_uncali)
  gyro = np.array(gyro)
  gyro_uncali = np.array(gyro_uncali)
  magn = np.array(magn)
  magn_uncali = np.array(magn_uncali)
  ahrs = np.array(ahrs)
  wifi = np.array(wifi)
  ibeacon = np.array(ibeacon)
  waypoint = np.array(waypoint)

  path_data_collection = PathDataCollection(acce, acce_uncali, gyro,
                                            gyro_uncali, magn, magn_uncali,
                                            ahrs, wifi, ibeacon, waypoint)
  sensor_df = create_sensor_df(path_data_collection)
  sensor_ts = sensor_df["ts"].values.astype(np.int64)

  return PathDataCollection(select(acce, sensor_ts),
                            select(acce_uncali, sensor_ts),
                            select(gyro, sensor_ts),
                            select(gyro_uncali, sensor_ts),
                            select(magn, sensor_ts),
                            select(magn_uncali, sensor_ts),
                            select(ahrs, sensor_ts), wifi, ibeacon, waypoint)


def select(sensor_arr: np.ndarray, ts: np.ndarray) -> np.ndarray:
  mask = np.isin(sensor_arr[:, 0].astype(np.int64), ts)
  return sensor_arr[mask, :]


def read_data_file(data_file_path: str) -> PathDataCollection:

  with open(data_file_path, 'r', encoding='utf-8') as file:
    return parse_data_text(file.readlines())


def create_sensor_df(path_data_collection: PathDataCollection) -> pd.DataFrame:
  magn_df = pd.DataFrame(path_data_collection.magn,
                         columns=["ts", "magn_x", "magn_y", "magn_z"])
  acce_df = pd.DataFrame(path_data_collection.acce,
                         columns=["ts", "acce_x", "acce_y", "acce_z"])
  ahrs_df = pd.DataFrame(path_data_collection.ahrs,
                         columns=["ts", "ahrs_x", "ahrs_y", "ahrs_z"])

  sensor_df = pd.merge(acce_df, magn_df, on="ts", how="outer")
  sensor_df.ffill(inplace=True)

  sensor_df = pd.merge(sensor_df, ahrs_df, on="ts", how="outer")
  sensor_df.ffill(inplace=True)

  sensor_df["ts"] = sensor_df["ts"].astype(np.int64)
  sensor_df.sort_values("ts", ascending=True, inplace=True)
  return sensor_df