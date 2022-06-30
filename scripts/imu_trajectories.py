from typing import Tuple
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from imu_sim import *


def checkerboard_traj_poses():

  # properties of checkerboard
  nsquares_width = 7
  nsquares_height = 6
  square_width = 0.05
  half_width = square_width * nsquares_width / 2
  half_height = square_width * nsquares_height / 2
  board_y = 0.25

  # Start
  qsb_start = np.array([0.0, 0.0, 0.0, 1.0])
  Tsb_start = np.zeros(3)

  # on the way to center top
  qsb_center_top_above = Rotation.from_euler('XYZ', [-75*np.pi/180, 0.0, 0.0]).as_quat()
  Tsb_center_top_above = np.array([0, 0.00, half_height+0.2])

  # Center, top
  qsb_center_top = Rotation.from_euler('XYZ', [np.pi/2, 0, np.pi]).as_quat()
  Tsb_center_top = np.array([0, board_y, half_height+0.175])

  # Start, but a little bit back
  qsb_start_back = qsb_start
  Tsb_start_back = np.array([0, -0.05, 0])

  # Start with 90 degrees cyclorotation and a little bit back
  qsb_start_90 = Rotation.from_euler('XYZ', [0.0, np.pi/2, 0.0]).as_quat()
  Tsb_start_90 = np.array([0, -0.05, 0])

  # Left side upright
  qsb_left_upright = Rotation.from_euler('XYZ', [0.0, 0.0, -np.pi/4]).as_quat()
  Tsb_left_upright = np.array([-half_width, 0.00, 0.0])

  # Left side upright 2
  qsb_left_upright2 = Rotation.from_euler('XYZ', [0.0, 0, -np.pi/3]).as_quat()
  Tsb_left_upright2 = np.array([-half_width-0.175, 0.0, 0.0])

  # Left side
  qsb_left = Rotation.from_euler('XYZ', [np.pi/2, 0, -np.pi/2]).as_quat()
  Tsb_left = np.array([ -half_width-0.175, board_y-0.05, 0])

  # Top left corner
  qsb_topleft = Rotation.from_euler('XYZ', [np.pi/2, 0, -3*np.pi/4]).as_quat()
  Tsb_topleft = np.array([ -half_width-0.05, board_y-0.05, half_height+0.05])

  qsb_topleft_above = Rotation.from_euler('XZX', [np.pi/2, -3*np.pi/4, -np.pi/4]).as_quat()
  Tsb_topleft_above = np.array([ -half_width-0.05, 0.00, half_height+0.05])

  qsb_topleft_diag = Rotation.from_euler('XZX', [np.pi/2, -3*np.pi/4, -np.pi/3]).as_quat()
  Tsb_topleft_diag = np.array([ -0.8*half_width, 0.00, 0.8*half_height])

  qsb_topright = Rotation.from_euler('XYZ', [np.pi/2, 0, 3*np.pi/4]).as_quat()
  Tsb_topright = np.array([ half_width+0.05, board_y, half_height+0.05])

  qsb_topright_above = Rotation.from_euler('XZX', [np.pi/2, 3*np.pi/4, -np.pi/4]).as_quat()
  Tsb_topright_above = np.array([ half_width+0.05, 0.00, half_height+0.05])

  qsb_topright_diag = Rotation.from_euler('XZX', [np.pi/2, 3*np.pi/4, -np.pi/3]).as_quat()
  Tsb_topright_diag = np.array([ 0.8*half_width, 0.00, 0.8*half_height])

  qsb_right_upright = Rotation.from_euler('XYZ', [0.0, 0.0, np.pi/4]).as_quat()
  Tsb_right_upright = np.array([half_width, 0.00, 0.0])

  qsb_right_upright2 = Rotation.from_euler('XYZ', [0.0, 0, np.pi/3]).as_quat()
  Tsb_right_upright2 = np.array([half_width+0.175, 0.0, 0.0])

  qsb_right = Rotation.from_euler('XYZ', [np.pi/2, 0, np.pi/2]).as_quat()
  Tsb_right = np.array([ half_width+0.175, board_y, 0])

  qsb_bottomright = Rotation.from_euler('XYZ', [np.pi/2, 0, np.pi/4]).as_quat()
  Tsb_bottomright = np.array([ half_width+0.05, board_y, -half_height-0.05])

  qsb_bottomright_above = Rotation.from_euler('XZX', [np.pi/2, np.pi/4, -np.pi/4]).as_quat()
  Tsb_bottomright_above = np.array([ half_width+0.05, 0.00, -half_height-0.05])

  qsb_bottomright_diag = Rotation.from_euler('XZX', [np.pi/2, np.pi/4, -np.pi/3]).as_quat()
  Tsb_bottomright_diag = np.array([ 0.8*half_width, 0.00, -0.8*half_height])

  qsb_center_bottom_above = Rotation.from_euler('XYZ', [75*np.pi/180, 0.0, 0.0]).as_quat()
  Tsb_center_bottom_above = np.array([0, 0.00, -half_height-0.2])

  qsb_center_bottom = Rotation.from_euler('XYZ', [np.pi/2, 0, 0]).as_quat()
  Tsb_center_bottom = np.array([ 0, board_y, -half_height-0.175])

  qsb_bottomleft = Rotation.from_euler('XYZ', [np.pi/2, 0, -np.pi/4]).as_quat()
  Tsb_bottomleft = np.array([ -half_width-0.05, board_y, -half_height-0.05])

  qsb_bottomleft_above = Rotation.from_euler('XZX', [np.pi/2, -np.pi/4, -np.pi/4]).as_quat()
  Tsb_bottomleft_above = np.array([ -half_width-0.05, 0.00, -half_height-0.05])

  qsb_bottomleft_diag = Rotation.from_euler('XZX', [np.pi/2, -np.pi/4, -np.pi/3]).as_quat()
  Tsb_bottomleft_diag = np.array([ -0.8*half_width, 0.00, -0.8*half_height])

  traj = np.vstack((
    np.hstack((0.0, qsb_start, Tsb_start)),
    np.hstack((1.0, qsb_start_back, Tsb_start_back)),
    np.hstack((2.0, qsb_start_90, Tsb_start_90)),
    np.hstack((3.0, qsb_start_back, Tsb_start_back)),
    np.hstack((4.0, qsb_start, Tsb_start)),
    np.hstack((6.0, qsb_left_upright, Tsb_left_upright)),
    np.hstack((8.0, qsb_left_upright2, Tsb_left_upright2)),
    np.hstack((10.0, qsb_left, Tsb_left)),
    np.hstack((12.0, qsb_topleft, Tsb_topleft)),
    np.hstack((14.0, qsb_topleft_above, Tsb_topleft_above)),
    np.hstack((16.0, qsb_topleft_diag, Tsb_topleft_diag)),
    np.hstack((18.0, qsb_start, Tsb_start)),
    np.hstack((20.0, qsb_center_top_above, Tsb_center_top_above)),
    np.hstack((22.0, qsb_center_top, Tsb_center_top)),
    np.hstack((24.0, qsb_center_top_above, Tsb_center_top_above)),
    np.hstack((26.0, qsb_topright_above, Tsb_topright_above)),
    np.hstack((28.0, qsb_topright, Tsb_topright)),
    np.hstack((30.0, qsb_topright_above, Tsb_topright_above)),
    np.hstack((32.0, qsb_topright_diag, Tsb_topright_diag)),
    np.hstack((34.0, qsb_start, Tsb_start)),
    np.hstack((36.0, qsb_right_upright, Tsb_right_upright)),
    np.hstack((38.0, qsb_right_upright2, Tsb_right_upright2)),
    np.hstack((40.0, qsb_right, Tsb_right)),
    np.hstack((42.0, qsb_bottomright_above, Tsb_bottomright_above)),
    np.hstack((44.0, qsb_bottomright, Tsb_bottomright)),
    np.hstack((46.0, qsb_bottomright_above, Tsb_bottomright_above)),
    np.hstack((48.0, qsb_bottomright_diag, Tsb_bottomright_diag)),
    np.hstack((50.0, qsb_start, Tsb_start)),
    np.hstack((52.0, qsb_center_bottom_above, Tsb_center_bottom_above)),
    np.hstack((54.0, qsb_center_bottom, Tsb_center_bottom)),
    np.hstack((56.0, qsb_center_bottom_above, Tsb_center_bottom_above)),
    np.hstack((58.0, qsb_bottomleft_above, Tsb_bottomleft_above)),
    np.hstack((60.0, qsb_bottomleft, Tsb_bottomleft)),
    np.hstack((62.0, qsb_bottomleft_above, Tsb_bottomleft_above)),
    np.hstack((64.0, qsb_bottomleft_diag, Tsb_bottomleft_diag)),
    np.hstack((66.0, qsb_start, Tsb_start))
  )).transpose()

  t_list = traj[0,:].flatten()
  qsb_vals = traj[1:5,:]
  Tsb_vals = traj[5:8,:]

  return (t_list, qsb_vals, Tsb_vals)



class LissajousSim(IMUSimBase):
  def __init__(self, **kwargs) -> None:
    IMUSimBase.__init__(self, **kwargs)

  def real_accel_gyro(self,
                      t: float,
                      X: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    Rsb = q2m(X[0:4])

    accel_x_s = -36*np.cos(3*t)
    accel_z_s = -16*np.sin(2*t)
    accel_y_s = -49*np.sin(7*t) / 10
    accel_s = np.array([accel_x_s, accel_y_s, accel_z_s])

    accel_b = Rsb.transpose() @ np.reshape(accel_s, (3,1))
    accel_b = accel_b.flatten()

    gyro = np.zeros(3)
    return (accel_b, gyro)


class TrefoilSim(IMUSimBase):
  def __init__(self, **kwargs) -> None:
    IMUSimBase.__init__(self, **kwargs)

  def real_accel_gyro(self,
                      t: float,
                      X: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    Rsb = q2m(X[0:4])

    accel_x_s = 12*np.sin(2*t)*np.sin(3*t) - 9*np.cos(2*t)*np.cos(3*t) - 4*np.cos(2*t)*(np.cos(3*t)+4)
    accel_y_s = -4*np.sin(2*t)*(np.cos(3*t)+4) - 12*np.cos(2*t)*np.sin(3*t) - 9*np.cos(3*t)*np.sin(2*t)
    accel_z_s = -9.0 * np.sin(3*t)
    accel_s = np.array([accel_x_s, accel_y_s, accel_z_s])

    accel_b = Rsb.transpose() @ np.reshape(accel_s, (3,1))
    accel_b = accel_b.flatten()

    #gyro_x = np.sin(0.3*t)
    #gyro_y = np.cos(0.4*t)
    #gyro_z = np.sin(0.1*t)
    #gyro = np.array([gyro_x, gyro_y, gyro_z])
    gyro = np.zeros(3)

    return (accel_b, gyro)



def get_imu_sim(motion_type: str,
                T: float=100.0,
                noise_accel: float=1e-4,
                noise_gyro: float=1e-5,
                bias_accel: np.ndarray=np.zeros(3),
                bias_gyro: np.ndarray=np.zeros(3),
                seed: int=None,
                grav_s: np.ndarray=np.array([0, 0, -9.8])
) -> IMUSimBase:
  if motion_type == "trefoil":
    return TrefoilSim(T=T, noise_accel=noise_accel, noise_gyro=noise_gyro,
                      bias_accel=bias_accel, bias_gyro=bias_gyro, seed=seed,
                      grav_s=grav_s, init_Vsb=np.array([0.0, 10.0, 3.0]))
  elif motion_type == "lissajous":
    return LissajousSim(T=T, noise_accel=noise_accel, noise_gyro=noise_gyro,
                        bias_accel=bias_accel, bias_gyro=bias_gyro, seed=seed,
                        grav_s=grav_s, init_Vsb=np.array([0.0, 0.7, 8]))
  elif motion_type == "checkerboard_traj":
    (t_list, qsb_vals, Tsb_vals) = checkerboard_traj_poses()
    return PoseInterpolationSim(t_list, qsb_vals, Tsb_vals, T=t_list[-1],
                                noise_accel=noise_accel, noise_gyro=noise_gyro,
                                bias_accel=bias_accel, bias_gyro=bias_gyro,
                                seed=seed,
                                grav_s=grav_s, init_Vsb=np.zeros(3))
  else:
    print("Unrecognized motion type, executing default IMU motion")
    return IMUSimBase(T=T, noise_accel=noise_accel, noise_gyro=noise_gyro,
                      bias_accel=bias_accel, bias_gyro=bias_gyro, seed=seed,
                      grav_s=grav_s, init_Vsb=np.zeros(3))



if __name__ == "__main__":

  traj_type = sys.argv[1]

  imu = get_imu_sim(traj_type)

  # Test retrieval
  if traj_type == "checkerboard_traj":
    Rsb, Tsb = imu.gsb(1.0)
    accel, gyro = imu.meas(0.5)
  else:
    Rsb, Tsb = imu.gsb(10)
    accel, gyro = imu.meas(20.0)

  # Test plotting
  imu.plt_ground_truth()

  plt.show()
