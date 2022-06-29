from typing import Tuple
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from imu_sim import *


def checkerboard_traj_poses():
  t_list = np.array([0.0, 5.0, 10.0])

  # properties of checkerboard
  nsquares_width = 7
  nsquares_height = 6
  square_width = 0.05
  half_width = square_width * nsquares_width / 2
  half_height = square_width * nsquares_height / 2
  board_y = 0.25

  # Start
  qsb0 = np.array([0.0, 0.0, 0.0, 1.0])
  Tsb0 = np.zeros(3)

  # Center, top
  qsb1 = Rotation.from_euler('XYZ', [np.pi/2, 0, np.pi]).as_quat()
  Tsb1 = np.array([0, board_y, half_height+0.175])

  # Start with 90 degrees cyclorotation
  qsb2 = Rotation.from_euler('XYZ', [0.0, np.pi, 0.0]).as_quat()
  Tsb2 = np.zeros(3)

  qsb_vals = np.vstack((qsb0, qsb1, qsb2)).transpose()
  Tsb_vals = np.vstack((Tsb0, Tsb1, Tsb2)).transpose()

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
