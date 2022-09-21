from typing import Union, List, Tuple

import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

def pinhole_project(Xc: np.ndarray,
            K: np.ndarray,
            imw: float,
            imh: float) -> Tuple[bool, np.ndarray]:
  if (Xc[2] <= 0):
    return (False, np.array([-1, -1]))
  else:
    xc = np.array([Xc[0] / Xc[2], Xc[1] / Xc[2], 1.0])
    xp = K[0:2,:] @ xc
    if (xp[0] < 0) or (xp[1] < 0) or (xp[0] > imw) or (xp[1] > imh):
      return (False, np.array([-1, -1]))
    else:
      return (True, xp)


class Point:
  def __init__(self, Xs: Union[List, np.ndarray, Tuple]):
    self.id = -1
    self.Xs = np.array(Xs)

  def Xc(self, Rsc: np.ndarray, Tsc: np.ndarray) -> np.ndarray:
    Xc = Rsc.transpose() @ (self.Xs - Tsc)
    return Xc

  def set_id(self, id_val: int) -> None:
    self.id = id_val

  def unset_id(self) -> None:
    self.id = -1

  def id_set(self) -> bool:
    return (self.id >= 0)


class PointCloudWorld:
  def __init__(self, seed=None):
    self.points = []
    self.next_pt_id = 10000   # constant `counter0` in src/feature.h
    self.rng = default_rng(seed)

  def addPt(self, Xs):
    self.points.append(Point(Xs))

  def writeMeasToFile(self, feature_ids, Xc_vals, xp_vals):
    with open("curr_measurements.txt", "w") as debug_file:
      for i,fid in enumerate(feature_ids):
        debug_file.write("{}: {} {} {} {} {}\n".format(
          fid, Xc_vals[i][0], Xc_vals[i][1], Xc_vals[i][2],
          xp_vals[i][0], xp_vals[i][1]
        ))

  def generateMeasurements(self,
                           gsc: np.ndarray,
                           K: np.ndarray,
                           imw: float,
                           imh: float,
                           noise_px_std: float):
    feature_ids = []
    xp_and_depths = []
    Xc_vals = []

    Rsc = gsc[:,:3]
    Tsc = gsc[:,3]

    for pt in self.points:
      Xc = pt.Xc(Rsc, Tsc)
      (is_visible, xp) = pinhole_project(Xc, K, imw, imh)
      if np.all(xp > 0):
        xp += noise_px_std * self.rng.standard_normal(2)
      if is_visible:
        # set the point's id if it wasn't set before
        if not pt.id_set():
          pt.set_id(self.next_pt_id)
          self.next_pt_id += 1
        xp_and_depths.append(np.array([xp[0], xp[1], Xc[2]]))
        feature_ids.append(pt.id)
        Xc_vals.append(Xc)
      else:
        if pt.id_set():
          pt.unset_id()

    feature_ids = np.array(feature_ids)
    xp_and_depths = np.array(xp_and_depths)
    #self.writeMeasToFile(feature_ids, Xc_vals, xp_vals)
    return (feature_ids, xp_and_depths)

  def plot_world(self):
    xs = np.zeros(len(self.points))
    ys = np.zeros(len(self.points))
    zs = np.zeros(len(self.points))
    for i,pt in enumerate(self.points):
      xs[i] = pt.Xs[0]
      ys[i] = pt.Xs[1]
      zs[i] = pt.Xs[2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


class RandomPCW(PointCloudWorld):
  def __init__(self, xlim, ylim, zlim, seed=None):
    PointCloudWorld.__init__(self, seed)
    self.xlim = xlim
    self.ylim = ylim
    self.zlim = zlim

  def addPt(self):
    x = self.rng.uniform(low=self.xlim[0], high=self.xlim[1])
    y = self.rng.uniform(low=self.ylim[0], high=self.ylim[1])
    z = self.rng.uniform(low=self.zlim[0], high=self.zlim[1])
    Xs = np.array([x, y, z])
    PointCloudWorld.addPt(self, Xs)

  def addNPts(self, N: int):
    for _ in range(N):
      self.addPt()


def letter_to_idx(letter: str):
  if letter == "x":
    idx = 0
  elif letter == "y":
    idx = 1
  else: # letter == "z"
    idx = 2
  return idx


class Checkerboard(PointCloudWorld):
  def __init__(self,
               square_width: float=0.05,
               dim_squares: Tuple[int, int]=(10,8),
               bot_right_coord: Tuple[float, float, float]=(10.0,10.0,10.0),
               plane: str="xz",
  ) -> None:
    PointCloudWorld.__init__(self)

    assert(plane in ["xz", "xy", "yz", "zx", "yx", "zy"])
    width_plane_idx = letter_to_idx(plane[0])
    height_plane_idx = letter_to_idx(plane[1])

    self.square_width = square_width
    self.dim_squares = dim_squares
    self.corner_pts_idx = []

    counter = 0
    for i in range(dim_squares[0]+1):
      for j in range(dim_squares[1]+1):
        Xs = np.array(bot_right_coord)
        Xs[width_plane_idx] += square_width*i
        Xs[height_plane_idx] += square_width*j
        self.addPt(Xs)

        if ((i==0) and (j==0)) or \
           ((i==0) and (j==dim_squares[1])) or \
           ((i==dim_squares[0]) and (j==0)) or \
           ((i==dim_squares[0]) and (j==dim_squares[1])):
          self.corner_pts_idx.append(counter) 
        counter += 1


  def allCornersVisible(self, gsc: np.ndarray, K: np.ndarray,
                        imw: float, imh: float
  ) -> List[ Tuple[bool, np.ndarray] ]:
    Rsc = gsc[:3,:3]
    Tsc = gsc[:3,3]

    output = []

    for idx in self.corner_pts_idx:
      pt = self.points[idx]
      Xc = pt.Xc(Rsc, Tsc)
      (is_visible, _) = pinhole_project(Xc, K, imw, imh)
      output.append((is_visible, pt.Xs, Xc))

    return output


if __name__ == "__main__":
  PCW = PointCloudWorld()
  PCW.addPt([10, 10, 1])
  PCW.addPt((10, 11, 1.5))

  RPCW = RandomPCW((-10, 10), (-10, 10), (-5, 5))
  RPCW.addNPts(100000)

  CPCW = Checkerboard(plane="xz", bot_right_coord=(9.75, 12.0, 0.80))

  Rsc = Rotation.from_euler('XYZ', [-np.pi/2, 0, 0]).as_matrix()
  Tsc = np.reshape(np.array([ 10, 8, 1 ]), (3, 1))
  gsc = np.hstack((Rsc, Tsc))
  K = np.array([[ 300,   0, 320 ],
                [   0, 300, 240 ],
                [   0,   0,   1 ]])

  (feature_ids0, xp0) = PCW.generateMeasurements(gsc, K, 640, 480)

  (feature_ids1, xp1) = RPCW.generateMeasurements(gsc, K, 640, 480)

  (feature_ids2, xp2) = CPCW.generateMeasurements(gsc, K, 640, 480)
  visible_corners = CPCW.allCornersVisible(gsc, K, 640, 480)

  import pdb
  pdb.set_trace()