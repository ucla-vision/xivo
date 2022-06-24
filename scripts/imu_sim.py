from typing import Tuple
import sys

import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation, Slerp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from pltutils import time_three_plots


D2R = np.pi / 180.0



# All quaternions use XYZW convention (same as scipy's quaternion library)


class Havertrig1d:
  def __init__(self, x0: float, x1: float, T: float) -> None:
    self.x0 = x0
    self.x1 = x1
    self.T = T

  def accel(self, t: float) -> np.ndarray:
    theta = t / self.T * np.pi

    # Haversine
    if (self.x1 > self.x0):
      accel = 0.5 * np.cos(theta) * (self.x1 - self.x0)
    # Havercosine
    else:
      accel = -0.5 * np.cos(theta) * (self.x0 - self.x1)

    # time-scale the acceleration
    accel_scaled = accel * (np.pi / self.T)**2
    return accel_scaled

  def vel(self, t: float) -> np.ndarray:
    theta = t / self.T * np.pi
    # Haversine
    if (self.x1 > self.x0):
      vel = 0.5 * np.sin(theta) * (self.x1 - self.x0)
    # Havercosine
    else:
      vel = -0.5 * np.sin(theta) * (self.x0 - self.x1)
    vel_scaled = vel * (np.pi / self.T)
    return vel_scaled

  def pos(self, t: float) -> np.ndarray:
    theta = t / self.T * np.pi
    # Haversine
    if (self.x1 > self.x0):
      pos = self.x0 + (self.x1 - self.x0)*(1 - np.cos(theta)) / 2
    # Havercosine
    else:
      pos = self.x1 + (self.x0 - self.x1)*(1 + np.cos(theta)) / 2
    return pos



class QuaternionSlew:
  def __init__(self, q0: np.ndarray, q1: np.ndarray, T: float) -> None:
    self.q0 = np.reshape(q0, 4)
    self.q1 = np.reshape(q1, 4)
    self.R = Rotation.from_quat(np.vstack([self.q0, self.q1]))
    self.T = T
    self.slerper = Slerp([0.0, self.T], self.R)

    q0x = self.q0[0]
    q0y = self.q0[1]
    q0z = self.q0[2]
    q0w = self.q0[3]

    q1x = self.q1[0]
    q1y = self.q1[1]
    q1z = self.q1[2]
    q1w = self.q1[3]

    # save some constants that came out of MATLAB symbolic toolbox
    self.sigma1 = np.arccos(q0w*q1w + q0x*q1x + q0y*q1y + q0z*q1z)
    self.sigma3 = q0w*q1y - q1w*q0y - q0x*q1z + q1x*q0z
    self.sigma4 = q0w*q1x - q0x*q1w + q0y*q1z - q0z*q1y
    self.sigma5 = q0w*q1z + q0x*q1y - q1w*q0z - q0y*q1x
    self.sigma2 = np.sqrt(self.sigma4**2 + self.sigma3**2 + self.sigma5**2)


  def slerp(self, t: float):
    return self.slerper(t).as_quat()

  def slerp_dot(self, t: float):
    tau = t / self.T

    q0x = self.q0[0]
    q0y = self.q0[1]
    q0z = self.q0[2]
    q0w = self.q0[3]

    q1x = self.q1[0]
    q1y = self.q1[1]
    q1z = self.q1[2]
    q1w = self.q1[3]

    # Symbolic expression generated using MATLAB Symbolic toolbox
    s1cos = self.sigma1 * np.cos(tau*self.sigma1)
    s1sin = np.sin(tau*self.sigma1) * self.sigma1

    dX = (q0w * s1cos * self.sigma4 / self.sigma2) \
         - (q0x * s1sin) \
         - (q0y * s1cos * self.sigma5 / self.sigma2) \
         + (q0z * s1cos * self.sigma3 / self.sigma2)

    dY = (q0w * s1cos * self.sigma3 / self.sigma2) \
         - (q0y * s1sin) \
         + (q0x * s1cos * self.sigma5 / self.sigma2) \
         - (q0z * s1cos * self.sigma4 / self.sigma2) \

    dZ = (q0w * s1cos * self.sigma5 / self.sigma2) \
         - (q0z * s1sin) \
         - (q0x * s1cos * self.sigma3 / self.sigma2) \
         + (q0y * s1cos * self.sigma4 / self.sigma2)

    dW = -(q0w * s1sin) \
         - (q0x * s1cos * self.sigma4 / self.sigma2) \
         - (q0y * s1cos * self.sigma3 / self.sigma2) \
         - (q0z * s1cos * self.sigma5 / self.sigma2)

    # time-scaled output
    slerp_dot = np.array([dX, dY, dZ, dW]) / self.T
    return slerp_dot


  def omega(self, t: float):
    q_t = self.slerp(t)
    qdot_t = self.slerp_dot(t)

    qx = q_t[0]
    qy = q_t[1]
    qz = q_t[2]
    qw = q_t[3]

    coeffMat = 0.5 * np.array([
      [  qw,  -qz,   qy ],
      [  qz,   qw,  -qx ],
      [ -qy,   qx,   qw ],
      #[ -qx,  -qy,  -qz ]
    ])
    #qdot_t = np.reshape(qdot_t, (4,1))
    #omega = np.linalg.pinv(coeffMat) @ qdot_t
    omega = np.linalg.solve(coeffMat, qdot_t[:3])
    omega = np.reshape(omega, 3)

    return omega



def cx(x: np.ndarray) -> np.ndarray:
  xhat = np.array([
    [     0,  x[2], -x[1]],
    [ -x[2],     0,  x[0]],
    [  x[1], -x[0],     0]
  ])
  return xhat


def qdot(q, omega):
  q = np.reshape(q, (4,1))

  omega_hat = cx(omega)
  omega_col = np.reshape(omega, (3,1))
  Omega0 = np.hstack((omega_hat, omega_col))
  Omega1 = np.hstack((-omega, 0))
  Omega = np.vstack((Omega0, Omega1))
  qdot = 0.5 * Omega @ q

  qdot = np.reshape(qdot, 4)
  return qdot


def q2m(q):
  return Rotation.from_quat(q).as_matrix()


def q2w(q):
  return Rotation.from_quat(q).as_rotvec()


class IMUSimBase:
  def __init__(self,
               T: float=100.0,
               noise_accel: float=1e-4,
               noise_gyro: float=1e-5,
               bias_accel: np.ndarray=np.zeros(3),
               bias_gyro: np.ndarray=np.zeros(3),
               seed: int=None,
               grav_s: np.ndarray=np.array([0, 0, -9.8]),
               init_Vsb: np.ndarray=np.zeros(3),
  ) -> None:
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

    output = solve_ivp(self.dX_dt, [0, self.T], ic, t_eval=np.arange(0.0, T, 0.001))
    self.t = output.t
    self.qsb = output.y[:4,:]
    self.Tsb = output.y[4:7,:]
    self.Vsb = output.y[7:10,:]
    self.interpolator = interp1d(self.t, output.y)

  def real_accel_gyro(self,
                      t: float,
                      X: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError

  def meas(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
    Rsb, _ = self.gsb(t)
    accel, gyro = self.real_accel_gyro(t, self.X(t))
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
    alpha_sb_b, omega_sb_b = self.real_accel_gyro(t, X)
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

  def X(self, t):
    return self.interpolator(t)

  def plt_ground_truth(self):
    Wsb = q2w(self.qsb.transpose()).transpose()
    time_three_plots(self.t, self.Tsb, "Ground truth Tsb (m)")
    time_three_plots(self.t, Wsb, "Ground truth Wsb (rad?)")



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


class PoseInterpolationSim(IMUSimBase):
  def __init__(self,
               t_list: np.ndarray,
               qsb_vals: np.ndarray,
               Tsb_vals: np.ndarray,
               T: float=100.0,
               noise_accel: float=1e-4,
               noise_gyro: float=1e-5,
               bias_accel: np.ndarray=np.zeros(3),
               bias_gyro: np.ndarray=np.zeros(3),
               seed: int=None,
               grav_s: np.ndarray=np.array([0, 0, -9.8]),
               init_Vsb: np.ndarray=np.zeros(3)
  ) -> None:
    """Simulator that moves the IMU from pose to pose. It slerps between
    orientations and haversines between positions. At each pose in `qsb_list`
    and `Tsb_list`, the IMU will be at rest (no angular velocity or linear
    acceleration). `t_list` is the absolute time for each pose.

    Dimensions of input numpy arrays are:
    - t_list: N
    - qsb_vals: (4, N)
    - Tsb_vals: (3, N)
    """
    # Save simulation parameters
    self.noise_accel = noise_accel
    self.noise_gyro = noise_gyro
    self.bias_accel = bias_accel
    self.bias_gyro = bias_gyro
    self.rng = default_rng(seed)
    self.grav_s = grav_s
    self.T = T

    # sanity checks on input
    N = len(t_list)
    assert(t_list.shape == (N,))
    assert(qsb_vals.shape == (4,N))
    assert(Tsb_vals.shape == (3,N))
    self.t_list = t_list
    self.qsb_vals = qsb_vals
    self.Tsb_vals = Tsb_vals

    # Add initial pose to the input list if it's not there
    if self.t_list[0] > 1e-6:
      self.t_list = np.hstack((0.0, t_list))
      init_qsb = np.reshape(np.array([0.0, 0.0, 0.0, 1.0]), (4,1))
      init_Tsb = np.zeros((3,1))
      self.qsb_vals = np.hstack((init_qsb, self.qsb_vals))
      self.Tsb_vals = np.hstack((init_Tsb, self.Tsb_vals))
      N += 1

    # set up pose interpolaters for each segment
    self.slerpers = []
    self.haversines_x = []
    self.haversines_y = []
    self.haversines_z = []
    for i in range(N-1):
      dT = self.t_list[i+1] - self.t_list[i]
      qslew = QuaternionSlew(self.qsb_vals[:,i], self.qsb_vals[:,i+1], dT)
      xslew = Havertrig1d(self.Tsb_vals[0,i], self.Tsb_vals[0,i+1], dT)
      yslew = Havertrig1d(self.Tsb_vals[1,i], self.Tsb_vals[1,i+1], dT)
      zslew = Havertrig1d(self.Tsb_vals[2,i], self.Tsb_vals[2,i+1], dT)
      self.slerpers.append(qslew)
      self.haversines_x.append(xslew)
      self.haversines_y.append(yslew)
      self.haversines_z.append(zslew)

    # Get pose without integration
    #output = solve_ivp(self.dX_dt, [0, self.T], ic, t_eval=np.arange(0.0, T, 0.001))
    self.t = np.arange(0.0, T, 0.001)
    self.qsb = np.zeros((4, self.t.size))
    self.Tsb = np.zeros((3, self.t.size))
    self.Vsb = np.zeros((3, self.t.size))
    for i,t in enumerate(self.t):
      X = self.X(t)
      self.qsb[:,i] = X[:4]
      self.Tsb[:,i] = X[4:7]
      self.Vsb[:,i] = X[7:10]


  def find_interval(self, t: float) -> int:
    if t == self.t_list[0]:
      return 0
    for i in range(self.t_list.size):
      if (self.t_list[i] < t) and (t <= self.t_list[i+1]):
        return i
    raise ValueError("PoseInterpolationSim: time {} is out of range".format(t))


  def real_accel_gyro(self,
                      t: float,
                      X: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    idx = self.find_interval(t)
    Rsb = q2m(X[0:4])

    time_into_interval = t - self.t_list[idx]

    accel_x_s = self.haversines_x[idx].accel(time_into_interval)
    accel_y_s = self.haversines_y[idx].accel(time_into_interval)
    accel_z_s = self.haversines_z[idx].accel(time_into_interval)
    accel_s = np.array([accel_x_s, accel_y_s, accel_z_s])

    accel_b = Rsb.transpose() @ accel_s

    gyro = self.slerpers[idx].omega(time_into_interval)

    return (accel_b, gyro)


  def gsb(self, t):
    idx = self.find_interval(t)
    time_into_interval = t - self.t_list[idx]

    qsb = self.slerpers[idx].slerp(time_into_interval)
    Tsb_x = self.haversines_x[idx].pos(time_into_interval)
    Tsb_y = self.haversines_y[idx].pos(time_into_interval)
    Tsb_z = self.haversines_z[idx].pos(time_into_interval)
    return (q2m(qsb), np.array([Tsb_x, Tsb_y, Tsb_z]))


  def X(self, t):
    idx = self.find_interval(t)
    time_into_interval = t - self.t_list[idx]

    qsb = self.slerpers[idx].slerp(time_into_interval)

    Tsb_x = self.haversines_x[idx].pos(time_into_interval)
    Tsb_y = self.haversines_y[idx].pos(time_into_interval)
    Tsb_z = self.haversines_z[idx].pos(time_into_interval)
    Tsb = np.array([Tsb_x, Tsb_y, Tsb_z])

    Vsb_x = self.haversines_x[idx].vel(time_into_interval)
    Vsb_y = self.haversines_y[idx].vel(time_into_interval)
    Vsb_z = self.haversines_z[idx].vel(time_into_interval)
    Vsb = np.array([Vsb_x, Vsb_y, Vsb_z])

    return np.hstack((qsb, Tsb, Vsb))


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
