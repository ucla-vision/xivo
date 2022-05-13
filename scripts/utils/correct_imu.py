# apply scale and misalignment correction to IMU measurement
# usage:
# python correct_imu.py --input /INPUT/CSV/FILE/OF/IMU/MEASUREMENTS --output /OUTPUT/CSV/FILE/OF/IMU/MEASUREMENTS
import numpy as np
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True,
        help='input CSV file of inertial measurements')
parser.add_argument('--output', type=str, required=True,
        help='output CSV file')
parser.add_argument('--gyro_calib', type=str, required=True,
        help="imu_tk gyro calibration output")
parser.add_argument('--accel_calib', type=str, required=True,
        help="imu_tk accel calibration output")
args = parser.parse_args()

def load_imu_intrinsics(filename, tag=''):
  ''' Load scaling, misalgnment matrix and bias vector from calibration results
  returned by imu_tk library.
  '''
  with open(filename, 'r') as fid:
    lines = fid.readlines()
    T = []  # misalignment
    for i in range(3):
      T.append([float(x) for x in lines[i].split()])
    T = np.array(T)
    # skip blank line
    K = []  # scaling
    for i in range(4, 7):
      K.append([float(x) for x in lines[i].split()])
    K = np.array(K)
    # skip blank line
    # bias
    b = np.array([float(x) for x in lines[8:11]])

  print("T{}=\n{}".format(tag, T))
  print("K{}=\n{}".format(tag, K))
  print("b{}=\n{}".format(tag, b))

  return T, K, b

if __name__ == '__main__':
  # load imu intrinsics
  Ta, Ka, ba = load_imu_intrinsics(args.accel_calib, 'a')
  Tg, Kg, bg = load_imu_intrinsics(args.gyro_calib, 'g')


  csv_filename =args.input
  output = args.output

  raw = []
  with open(csv_filename, 'r') as fid:
    # skip the header
    head = fid.readline()
    for line in fid.readlines():
      data = line.split(',')
      ts = int(data[0])
      gyro = np.array([float(x) for x in data[1:4]])
      accel = np.array([float(x) for x in data[4:7]])

      if np.linalg.norm(gyro) == 0 or np.linalg.norm(accel) == 0:
          raise ValueError('empty gyro or accel! need interpolation first?')

      raw.append((ts, gyro, accel))


  if len(raw) > 0:
    corrected = []
    for ts, gyro, accel in raw:

      calib_gyro = Tg.dot(Kg.dot(gyro + bg))
      calib_accel = Ta.dot(Ka.dot(accel + ba))

      corrected.append((ts, calib_gyro, calib_accel))

    with open(output, 'w') as ofid:
      ofid.write(head)
      for ts, gyro, accel in corrected:
        ofid.write('{},{},{},{},{},{},{}\n'.format(ts, gyro[0], gyro[1], gyro[2], accel[0], accel[1], accel[2]))

  else:
    raise ValueError('empty list of raw measurement!')
