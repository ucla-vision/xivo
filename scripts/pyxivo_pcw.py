import argparse
from distutils.command.clean import clean
import json
import os, glob

import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'lib')
import pyxivo
from point_cloud_world import RandomPCW, Checkerboard
from imu_sim import get_imu_sim
from utils import cleanup_and_load_json


parser = argparse.ArgumentParser()
parser.add_argument("-npts", default=1000, type=int)
parser.add_argument("-xlim", default=[-10, 10], nargs=2, type=float)
parser.add_argument("-ylim", default=[-10, 10], nargs=2, type=float)
parser.add_argument("-zlim", default=[-5, 5], nargs=2, type=float)
parser.add_argument("-pcw_seed", default=0, type=int)
parser.add_argument("-imu_seed", default=1, type=int)
parser.add_argument("-noise_accel", default=1e-4, type=float)
parser.add_argument("-noise_gyro", default=1e-5, type=float)
parser.add_argument("-motion_type", default="sinusoid", type=str)
parser.add_argument("-total_time", default=100.0, type=float)
parser.add_argument("-imu_dt", default=0.0025, type=float)
parser.add_argument("-vision_dt", default=0.04, type=float)
parser.add_argument('-cfg', type=str, default="cfg/pcw.json")
parser.add_argument('-viewer_cfg', type=str, default="cfg/pcw_viewer.json")
parser.add_argument('-use_viewer', default=False, action="store_true")
parser.add_argument('-tracker_only', default=False, action="store_true")


def read_cfg_data(cfg_json: str):
  # load file and strip comments
  cfg_data = cleanup_and_load_json(cfg_json)

  # Rbc, Tbc
  X = cfg_data["X"]
  Wbc = np.array(X["Wbc"])
  Tbc = np.reshape(np.array(X["Tbc"]), (3,1))
  if Wbc.size == 3:
    Rbc = Rotation.from_rotvec(Wbc).as_matrix()
  else:
    Rbc = Wbc

  # Camera intrinsics
  camera_cfg = cfg_data["camera_cfg"]
  assert(camera_cfg["model"] == "pinhole")
  fx = camera_cfg["fx"]
  fy = camera_cfg["fy"]
  cx = camera_cfg["cx"]
  cy = camera_cfg["cy"]
  K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]], dtype=np.float64)
  imw = camera_cfg["cols"]
  imh = camera_cfg["rows"]

  # gravity
  grav_g = np.reshape(np.array(cfg_data["gravity"]), (3,1))
  Rsg = Rotation.from_rotvec(np.array(X["Wsg"])).as_matrix()
  grav_s = Rsg @ grav_g

  return (Rbc, Tbc, K, imw, imh, grav_s)


def main(args):
  # Read Wbc, Tbc from .cfg file
  Rbc, Tbc, K, imw, imh, grav_s = read_cfg_data(args.cfg)

  imu = get_imu_sim(args.motion_type,
                    T=args.total_time,
                    noise_accel=args.noise_accel,
                    noise_gyro=args.noise_gyro,
                    seed=args.imu_seed,
                    grav_s=grav_s)
  if args.motion_type == "checkerboard_traj":
    nsquares_width = 7
    nsquares_height = 6
    square_width = 0.05
    half_width = square_width * nsquares_width / 2
    half_height = square_width * nsquares_height / 2
    board_y = 0.25
    vision = \
      Checkerboard(square_width=square_width,
                   dim_squares=(nsquares_width, nsquares_height),
                   bot_right_coord=(-half_width, board_y, -half_height),
                   plane="xz")
  else:
    vision = RandomPCW(args.xlim, args.ylim, args.zlim, seed=args.pcw_seed)
    vision.addNPts(args.npts)

  # Assemble IMU and vision packets in order of arrival. If two packets have the
  # same timestamp, then the IMU packet arrives first
  if args.tracker_only:
    imu_meas_times = np.zeros(0)
    imu_meas = np.zeros((2,0))
  else:
    imu_meas_times = np.arange(0, args.total_time, args.imu_dt) # (NT,)
    imu_meas = np.vstack((imu_meas_times, np.zeros(imu_meas_times.size))) # (2, NT)
  vision_meas_times = np.arange(0, args.total_time, args.vision_dt) # (NT,)
  vision_meas = np.vstack((vision_meas_times, np.ones(vision_meas_times.size))) # (2, NT)
  all_packets = np.hstack((imu_meas, vision_meas))  # (2, 2*NT)
  all_packets = all_packets[:,all_packets[0,:].argsort()]

  # Lambda function: whether or not a packet is IMU 
  is_imu = lambda x: (x[1] < 0.5)

  # estimator object
  estimator = pyxivo.Estimator(args.cfg, args.viewer_cfg, args.motion_type,
                               args.tracker_only)
  for i in range(all_packets.shape[1]):
    packet = all_packets[:,i]
    t = all_packets[0,i]
    if is_imu(packet):
      accel, gyro = imu.meas(t)
      estimator.InertialMeas(int(t*1e9), gyro[0], gyro[1], gyro[2], accel[0],
                             accel[1], accel[2])
    else:
      Rsb, Tsb = imu.gsb(t)
      Rsc = Rsb @ Rbc
      Tsc = Rsb @ Tbc + np.reshape(Tsb, (3,1))
      gsc = np.hstack((Rsc, Tsc))
      (feature_ids, xp_vals) = vision.generateMeasurements(gsc, K, imw, imh)
      if len(feature_ids) > 0:
        if args.tracker_only:
          estimator.VisualMeasPointCloudTrackerOnly(int(t*1e9), feature_ids, xp_vals)
        else:
          estimator.VisualMeasPointCloud(int(t*1e9), feature_ids, xp_vals)
      estimator.Visualize()


if __name__ == "__main__":
  args = parser.parse_args()
  main(args)