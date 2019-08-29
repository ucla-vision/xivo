# Perform linear interpolation on asynchronized inertial measurements.
# usage:
# python interpolate_imu.py --input /INPUT/CSV/FILE/OF/IMU/MEASUREMENTS --output /OUTPUT/CSV/FILE/OF/IMU/MEASUREMENTS
import numpy as np
import matplotlib.pyplot as plt
import argparse, sys, os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True,
        help='input CSV file of inertial measurements')
parser.add_argument('--output', type=str, required=True,
        help='output CSV file')
parser.add_argument('--check-ts', action='store_true', default=False,
        help='if set, check timestamps')
args = parser.parse_args()

def interpolate(now, prev_m, next_m):
  t0, m0 = prev_m
  t1, m1 = next_m
  assert now > t0 and now < t1
  duration = t1 - t0
  return m0 * (t1 - now) / duration + m1 * (now - t0) / duration

def find_next(curr, data, field):
  if curr >= len(data):
    return None

  for i, (ts, gyro, accel) in enumerate(data[curr + 1:]):
    if field == 'gyro' and np.linalg.norm(gyro) > 0:
      return (ts, gyro)

    if field == 'accel' and np.linalg.norm(accel) > 0:
      return (ts, accel)

  return None

def find_next_gyro(curr, data):
  return find_next(curr, data, field='gyro')

def find_next_accel(curr, data):
  return find_next(curr, data, field='accel')




if __name__ == '__main__':

  csv_filename =args.input
  output = args.output

  raw = []
  with open(csv_filename, 'r') as fid:
    # skip the header
    head = fid.readline()
    for line in fid.readlines():
      data = line.split(',')
      ts = int(data[0])
      # gyro = np.array([float(x) for x in data[1:4]])
      # accel = np.array([float(x) for x in data[4:]])

      gyro = np.array([float(x) for x in data[1:4]])
      accel = np.array([float(x) for x in data[4:7]])

      raw.append((ts, gyro, accel))


  if len(raw) > 0:
    interp = []
    prev_gyro = None
    prev_accel = None
    for i, (ts, gyro, accel) in enumerate(raw):
      if np.linalg.norm(gyro) == 0 and np.linalg.norm(accel) == 0:
        raise ValueError('both gyro and accel measurements are empty!')

      if np.linalg.norm(gyro) == 0:
        if prev_gyro is None:
          prev_gyro = (ts, gyro)
        else:
          next_gyro = find_next_gyro(i, raw)
          if next_gyro is not None:
            gyro = interpolate(ts, prev_gyro, next_gyro)
            interp.append((ts, gyro, accel))
        # accel must not be empty, update
        prev_accel = (ts, accel)
      elif np.linalg.norm(accel) == 0:
        if prev_accel is None:
          prev_accel = (ts, accel)
        else:
          next_accel = find_next_accel(i, raw)
          if next_accel is not None:
            accel = interpolate(ts, prev_accel, next_accel)
            interp.append((ts, gyro, accel))
        # gyro must not be empty, update
        prev_gyro = (ts, gyro)
      else:
        # both fields are non-empty, forward
        prev_gyro, prev_accel = gyro, accel
        interp.append((ts, gyro, accel))

    with open(output, 'w') as ofid:
      ofid.write(head)
      for ts, gyro, accel in interp:
        ofid.write('{},{},{},{},{},{},{}\n'.format(ts, gyro[0], gyro[1], gyro[2], accel[0], accel[1], accel[2]))

    if args.check_ts:
      ts = [triplet[0] for triplet in interp]
      dt = np.diff(ts) * 1e-6
      plt.plot(dt, '.')
      plt.show()

  else:
    raise ValueError('empty list of raw measurement!')