from typing import Tuple

import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from pltutils import time_three_plots


D2R = np.pi / 180.0


def cx(x: np.ndarray) -> np.ndarray:
  xhat = np.array([
    [     0,  x[2], -x[1]],
    [ -x[2],     0,  x[0]],
    [  x[1], -x[0],     0]
  ])
  return xhat


def qdot(q, omega):
  omega_hat = cx(omega)
  omega_col = np.reshape(omega, (3,1))
  Omega0 = np.hstack((0, -omega))
  Omega1 = np.hstack((omega_col, omega_hat))
  Omega = np.vstack((Omega0, Omega1))
  return 0.5 * Omega @ q


def q2m(q):
  return Rotation.from_quat(q).as_matrix()


def q2w(q):
  return Rotation.from_quat(q).as_rotvec()


class IMUSim:
  def __init__(self,
               type: str,
               T: float=100.0,
               noise_accel: float=1e-4,
               noise_gyro: float=1e-5,
               bias_accel: np.ndarray=np.zeros(3),
               bias_gyro: np.ndarray=np.zeros(3),
               seed: int=None,
               grav_s: np.ndarray=np.array([0, 0, -9.8]),
               init_Vsb: np.ndarray=np.zeros(3),
  ) -> None:
    assert(type in ["sinusoid", "straight", "VR", "box"])
    self.type = type
    self.noise_accel = noise_accel
    self.noise_gyro = noise_gyro
    self.bias_accel = bias_accel
    self.bias_gyro = bias_gyro
    self.rng = default_rng(seed)
    self.grav_s = grav_s
    self.T = T

    # solve for ground-truth values
    init_qsb = np.array([0, 0, 0, 1])
    init_Tsb = np.zeros(3,)
    ic = np.hstack((init_qsb, init_Tsb, init_Vsb))
    output = solve_ivp(self.dX_dt, [0, self.T], ic)
    self.t = output.t
    self.qsb = output.y[:4,:]
    self.Tsb = output.y[4:7,:]
    self.Vsb = output.y[7:10,:]
    self.interpolator = interp1d(self.t, output.y)


  def sinusoid_meas(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
    accel_x = 0.2 * np.sin(3 * t)
    accel_y = 0.1 * np.cos(0.4 * t)
    accel_z = 0.1 * np.sin(5 * t)
    accel = np.array([accel_x, accel_y, accel_z])
    gyro_x = -0.5 * D2R * np.sin(0.3 * t)
    gyro_y = 0.5 * D2R * np.cos(0.1 * t)
    gyro_z = 0.3 * D2R * np.sin(0.1 * t)
    gyro = np.array([gyro_x, gyro_y, gyro_z])
    return (accel, gyro)

  def straight_line_meas(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError

  def VR_meas(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError

  def box_meas(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError

  def real_accel_gyro(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
    if self.type == "sinusoid":
      accel, gyro = self.sinusoid_meas(t)
    elif self.type == "straight":
      accel, gyro = self.straight_line_meas(t)
    elif self.type == "VR":
      accel, gyro = self.VR_meas(t)
    elif self.type == "box":
      accel, gyro = self.box_meas(t)
    return accel, gyro

  def meas(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
    Rsb, _ = self.gsb(t)
    accel, gyro = self.real_accel_gyro(t)
    noise_a = self.noise_accel * self.rng.standard_normal(3)
    noise_g = self.noise_gyro * self.rng.standard_normal(3)  
    meas_a = accel + noise_a + self.bias_accel - np.squeeze(Rsb @ self.grav_s)
    meas_g = gyro + noise_g + self.bias_gyro
    return meas_a, meas_g

  def dX_dt(self, t, X):
    curr_qsb = X[0:4]
    curr_Rsb = q2m(curr_qsb) 
    curr_Vsb = X[7:10]

    # these are alpha_sb_b and omega_sb_b
    alpha_sb_b, omega_sb_b = self.real_accel_gyro(t)
    alpha_sb_b_col = np.reshape(alpha_sb_b, (3,1))

    Xdot = np.zeros(10)
    Xdot[0:4] = qdot(curr_qsb, omega_sb_b)
    Xdot[4:7] = curr_Vsb
    Xdot[7:10] = np.squeeze(curr_Rsb @ alpha_sb_b_col)
    return Xdot

  def gsb(self, t):
    X_t = self.interpolator(t)
    qsb = X_t[:4]
    Tsb = X_t[4:7]
    return (q2m(qsb), Tsb)

  def plt_ground_truth(self):
    Wsb = q2w(self.qsb.transpose()).transpose()
    time_three_plots(self.t, self.Tsb, "Ground truth Tsb (m)")
    time_three_plots(self.t, Wsb, "Ground truth Wsb (rad?)")



if __name__ == "__main__":
  imu = IMUSim("sinusoid")

  # Test retrieval
  Rsb, Tsb = imu.gsb(10)
  accel, gyro = imu.meas(20.0)

  # Test plotting
  imu.plt_ground_truth()

  plt.show()