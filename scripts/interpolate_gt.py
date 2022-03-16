import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse


parser = argparse.ArgumentParser("Interpolates and plots ground truth data")
parser.add_argument('-dump', default='.',
  help='place where data was saved')
parser.add_argument('-dataset', default='tumvi', help='name of dataset')
parser.add_argument('-seq', default='room1',
  help='name of sequence within dataset')
parser.add_argument('-sen', default='tango_top',
  help='sensor id, for datasets with multiple types of sensors')
parser.add_argument('-cam_id', default=0,
  help='camera id, if using a stereo dataset')



def interpolate_gsb(timestamps, R_sb_list, T_sb_list, V_sb_list, query_timestamp):
    # Interpolate translation
    T_sb_list = np.array(T_sb_list)
    interp_T = interpolate.interp1d(timestamps, T_sb_list, axis=0)
    T_sb = interp_T(np.float(query_timestamp))

    # Interpolate velocity
    V_sb_list = np.array(V_sb_list)
    interp_V = interpolate.interp1d(timestamps, V_sb_list, axis=0)
    V_sb = interp_V(np.float(query_timestamp))

    # Interpolate Rotation (slerp)
    slerp = Slerp(timestamps, R_sb_list)
    R_sb = slerp(np.array(query_timestamp))

    return (R_sb, T_sb, V_sb)


class groundtruth_interpolator:
  def __init__(self, gt_filename):
    self.gt_timestamps_int = []
    self.gt_timestamps_float = []
    self.Rsb = []
    self.Tsb = []
    self.Vsb = []

    self.load_gt_file(gt_filename)
    self.compute_velocities()


  def load_gt_file(self, gt_filename):
    gt_file = open(gt_filename, "r")

    gt_timestamps = []

    for line in gt_file:
        linearr = line.split()
        timestamp_sec = float(linearr[0])
        timestamp_ns = int(timestamp_sec * 1e9)
        T_sb = np.array([float(linearr[1]),
                         float(linearr[2]),
                         float(linearr[3])])
        q_sb_xyzw = np.array([float(linearr[4]),
                              float(linearr[5]),
                              float(linearr[6]),
                              float(linearr[7])])
        R_sb = Rotation.from_quat(q_sb_xyzw)
        gt_timestamps.append(timestamp_ns)
        self.Rsb.append(R_sb)
        self.Tsb.append(T_sb)

    self.gt_timestamps_int = np.array(gt_timestamps, dtype=np.int)
    self.gt_timestamps_float = np.array(gt_timestamps, dtype=np.float)
    self.Tsb = np.array(self.Tsb)


  def compute_velocities(self):
    """estimate velocities using central difference formula."""

    num_timestamps = len(self.gt_timestamps_float)
    for idx in range(num_timestamps):
      T_sb = self.Tsb[idx,:]
      if idx==0:
        T_sb1 = self.Tsb[idx+1,:]
        dt = self.gt_timestamps_float[idx+1] - self.gt_timestamps_float[idx]
        V_sb = (T_sb1 - T_sb) / dt
      elif idx==num_timestamps-1:
        T_sb0 = self.Tsb[idx-1,:]
        dt = self.gt_timestamps_float[idx] - self.gt_timestamps_float[idx-1]
        V_sb = (T_sb - T_sb0) / dt
      else:
        T_sb1 = self.Tsb[idx+1,:]
        T_sb0 = self.Tsb[idx-1,:]
        dt = self.gt_timestamps_float[idx+1] - self.gt_timestamps_float[idx-1]
        V_sb = (T_sb1 - T_sb0) / dt * 1e9
      self.Vsb.append(V_sb)

    self.Vsb = np.array(self.Vsb)


  def get_gt_gsb(self, query_timestamp):
    idx = np.abs(self.gt_timestamps_int - query_timestamp).argmin()
    closest_gt_timestamp = self.gt_timestamps_int[idx]
    R_sb_0 = self.Rsb[idx]
    T_sb_0 = self.Tsb[idx,:]
    V_sb_0 = self.Vsb[idx,:]

    if (query_timestamp == closest_gt_timestamp) or (idx==0) or (idx==len(self.gt_timestamps_int)-1):
        R_sb = R_sb_0
        T_sb = T_sb_0
        V_sb = V_sb_0
    elif query_timestamp > closest_gt_timestamp:
        other_gt_timestamp = self.gt_timestamps_int[idx+1]
        R_sb_1 = self.Rsb[idx+1]
        T_sb_1 = self.Tsb[idx+1,:]
        V_sb_1 = self.Vsb[idx+1,:]
        (R_sb, T_sb, V_sb) = interpolate_gsb([closest_gt_timestamp, other_gt_timestamp],
                                             Rotation.from_quat([
                                                 R_sb_0.as_quat(),
                                                 R_sb_1.as_quat()
                                             ]),
                                             [T_sb_0, T_sb_1],
                                             [V_sb_0, V_sb_1],
                                             query_timestamp)
    else:
        other_gt_timestamp = self.gt_timestamps_int[idx-1]
        R_sb_1 = self.Rsb[idx-1]
        T_sb_1 = self.Tsb[idx-1,:]
        V_sb_1 = self.Vsb[idx-1,:]
        (R_sb, T_sb, V_sb) = interpolate_gsb([other_gt_timestamp, closest_gt_timestamp],
                                             Rotation.from_quat([
                                                 R_sb_1.as_quat(),
                                                 R_sb_0.as_quat()
                                             ]),
                                             [T_sb_1, T_sb_0],
                                             [V_sb_0, V_sb_1],
                                             query_timestamp)
    return (R_sb, T_sb, V_sb)


  def subfigure_plot(self, plotitem='Translation'):
    if plotitem=='Translation':
      plotvals = self.Tsb
      title = 'Ground Truth Translation (m)'
    elif plotitem=='Velocity':
      plotvals = self.Vsb
      title = 'Ground Truth Velocity (m/s)'
    xvals = (self.gt_timestamps_float - self.gt_timestamps_float[0])*1e-9

    plt.figure()
    plt.suptitle(title)
    plt.subplot(3,1,1)
    plt.plot(xvals, plotvals[:,0])
    plt.ylabel('x')
    plt.subplot(3,1,2)
    plt.plot(xvals, plotvals[:,1])
    plt.ylabel('y')
    plt.subplot(3,1,3)
    plt.plot(xvals, plotvals[:,2])
    plt.ylabel('z')
    plt.xlabel('Time (s)')


  def traj_plot_3d(self):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(self.Tsb[:,0], self.Tsb[:,1], self.Tsb[:,2])
    ax.scatter(self.Tsb[0,0], self.Tsb[0,1], self.Tsb[0,2], c='r')
    ax.scatter(self.Tsb[-1,0], self.Tsb[-1,1], self.Tsb[-1,2], c='g')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')


if __name__=="__main__":
  args = parser.parse_args()

  if args.dataset in ["tumvi", "euroc"]:
    gt_data = os.path.join(args.dump, '{}_{}_gt'.format(args.dataset, args.seq))
  elif args.dataset=="cosyvio":
    gt_data = os.path.join(args.dump,
      'cosyvio_{}_{}_gt'.format(args.sen, args.seq))

  obj = groundtruth_interpolator(gt_data)
  obj.subfigure_plot(plotitem='Translation')
  obj.subfigure_plot(plotitem='Velocity')
  obj.traj_plot_3d()

  plt.show()
