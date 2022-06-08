from typing import Union, List, Tuple

import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation


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
  def __init__(self):
    self.points = []
    self.points_visible = []
    self.next_pt_id = 10000   # constant `counter0` in src/feature.h

  def addPt(self, Xs):
    self.points.append(Point(Xs))

  def generateMeasurements(self,
                           gsc: np.ndarray,
                           K: np.ndarray,
                           imw: float,
                           imh: float):
    feature_ids = []
    xp_vals = []

    Rsc = gsc[:,:3]
    Tsc = gsc[:,3]

    for pt in self.points:
      Xc = pt.Xc(Rsc, Tsc)
      (is_visible, xp) = pinhole_project(Xc, K, imw, imh)
      if is_visible:
        # set the point's id if it wasn't set before
        if not pt.id_set():
          pt.set_id(self.next_pt_id)
          self.next_pt_id += 1
        xp_vals.append(xp)
        feature_ids.append(pt.id)

      else:
        if pt.id_set():
          pt.unset_id()

    feature_ids = np.array(feature_ids)
    xp_vals = np.array(xp_vals)
    return (feature_ids, xp_vals)


class RandomPCW(PointCloudWorld):
  def __init__(self, xlim, ylim, zlim, seed=None):
    PointCloudWorld.__init__(self)
    self.xlim = xlim
    self.ylim = ylim
    self.zlim = zlim
    self.rng = default_rng(seed)

  def addPt(self):
    x = self.rng.uniform(low=self.xlim[0], high=self.xlim[1])
    y = self.rng.uniform(low=self.ylim[0], high=self.ylim[1])
    z = self.rng.uniform(low=self.zlim[0], high=self.zlim[1])
    Xs = np.array([x, y, z])
    PointCloudWorld.addPt(self, Xs)

  def addNPts(self, N: int):
    for _ in range(N):
      self.addPt()


if __name__ == "__main__":
  PCW = PointCloudWorld()
  PCW.addPt([10, 10, 1])
  PCW.addPt((10, 11, 1.5))

  RPCW = RandomPCW((-10, 10), (-10, 10), (-5, 5))
  RPCW.addNPts(100000)

  Rsc = Rotation.from_euler('XYZ', [-np.pi/2, 0, 0]).as_matrix()
  Tsc = np.reshape(np.array([ 10, 8, 1 ]), (3, 1))
  gsc = np.hstack((Rsc, Tsc))
  K = np.array([[ 300,   0, 320 ],
                [   0, 300, 240 ],
                [   0,   0,   1 ]])

  (feature_ids0, xp0) = PCW.generateMeasurements(gsc, K, 640, 480)

  (feature_ids1, xp1) = RPCW.generateMeasurements(gsc, K, 640, 480)

  import pdb
  pdb.set_trace()