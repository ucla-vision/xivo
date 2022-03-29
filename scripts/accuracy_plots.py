import argparse

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from estimator_data import EstimatorData
from interpolate_gt import groundtruth_interpolator
from pltutils import error_three_plots, plot_3D_error_cloud, time_plot, \
  plot_3d_trajectories, time_three_plots, matrix_frodiff_plot, time_n_plots
from utils import rigid_transform_3d, get_xivo_gt_filename, \
  get_xivo_output_filename


parser = argparse.ArgumentParser()
parser.add_argument("-dump", default=".",
  help="location of xivo's output data from a dataset")
parser.add_argument("-dataset", default="tumvi",
  help="name of a (supported) VIO dataset [tumvi|cosyvio|euroc|xivo]")
parser.add_argument("-seq", default="room1",
  help="short tag for sequence name")
parser.add_argument("-cam_id", default=0, type=int,
  help="camera from stereo camera pair (only used for tumvi dataset)")
parser.add_argument("-sen", default="tango_top",
  help="sensor from which images were captured (only used for cosyvio dataset)")



class PlotHelper:
  def __init__(self, seq, gt_file, estimator_results,
               start_ind=0, end_ind=None):

    self.seq = seq

    # Text for x-axis time plots
    self.time_axis_label = 'Time (s)'

    # Load ground truth data
    self.gt_interpolator = groundtruth_interpolator(gt_file)

    # load estimator data
    self.est = EstimatorData(estimator_results, start_ind=start_ind,
      end_ind=end_ind)

    self.time_axis = np.zeros((self.est.nposes,))
    self.time_axis_orig = []

    # Ground truth and error trajectories
    self.Rsb_gt = []
    self.Tsb_gt = np.zeros((3,self.est.nposes))
    self.Vsb_gt = np.zeros((3,self.est.nposes))
    self.Wsb_gt = np.zeros((3,self.est.nposes)) # duplicate of self.Rsb_gt

    self.Rsb_error = []
    self.Wsb_error = np.zeros((3,self.est.nposes))
    self.Tsb_error = np.zeros((3,self.est.nposes))
    self.Vsb_error = np.zeros((3,self.est.nposes))


  def align_gt_to_est(self):
    for i in range(self.est.nposes):
      timestamp = self.est.time_axis[i]

      R_sb_gt, T_sb_gt, V_sb_gt = self.gt_interpolator.get_gt_gsb(timestamp)
      self.Rsb_gt.append(R_sb_gt)
      self.Tsb_gt[:,i] = T_sb_gt
      self.Vsb_gt[:,i] = V_sb_gt
      self.Wsb_gt[:,i] = R_sb_gt.as_rotvec()

      self.time_axis[i] = timestamp

    self.time_axis = np.linspace(0,self.est.nposes-1,self.est.nposes)

    # copy time axis before we mess with it
    self.time_axis_orig = self.time_axis

    # Align the ground truth and estimator data
    rot,trans = rigid_transform_3d(np.transpose(self.Tsb_gt),
                                    np.transpose(self.est.Tsb))
    rotObj = Rotation.from_matrix(rot)
    for i in range(self.est.nposes):
      self.Rsb_gt[i] = rotObj * self.Rsb_gt[i]
      self.Tsb_gt[:,i] = rot.dot(self.Tsb_gt[:,i]) + trans
      self.Vsb_gt[:,i] = rot.dot(self.Vsb_gt[:,i])


  def compute_errors(self):
    for ind in range(self.est.nposes):
      self.Tsb_error[:,ind] = self.est.Tsb[:,ind] - self.Tsb_gt[:,ind]
      self.Vsb_error[:,ind] = self.est.Vsb[:,ind] - self.Vsb_gt[:,ind]
      self.Rsb_error.append(self.est.Rsb[ind] * self.Rsb_gt[ind].inv())
      self.Wsb_error[:,ind] = self.Rsb_error[-1].as_rotvec().flatten()


  def show_state_errors(self):
    print("\nState Errors Display:")
    Tsb_error = self.est.Tsb - self.Tsb_gt
    vel_error = self.est.Vsb - self.Vsb_gt
    rot_error = np.zeros((3,self.est.nposes))
    for ind in range(self.est.nposes):
      R_sb_diff = self.est.Rsb[ind] * self.Rsb_gt[ind].inv()
      rot_error[:,ind] = R_sb_diff.as_rotvec().flatten()

    error_three_plots(self.time_axis, rot_error, self.seq, "rotation", "rad?")
    error_three_plots(self.time_axis, Tsb_error, self.seq, "translation", "m")
    error_three_plots(self.time_axis, vel_error, self.seq, "velocity", "m/s")
    plot_3D_error_cloud(rot_error, "Rotation Error Cloud (rad?)")
    plot_3D_error_cloud(Tsb_error, "Translation Error Cloud (m)")
    plot_3D_error_cloud(vel_error, "Velocity Error Cloud (m/s)")


  def plot_gauge_group(self):
    time_plot(self.time_axis_orig, self.est.gauge_group,
      title="Reference Group ID", xlabel=self.time_axis_label)


  def plot_trajectories(self, npts=-1):
    plot_3d_trajectories(self.est.Tsb, self.Tsb_gt)


if __name__ == "__main__":
  args = parser.parse_args()

  # get data files
  estimator_data = get_xivo_output_filename(args.dump, args.dataset, args.seq,
    cam_id=args.cam_id, sen=args.sen)
  gt_data = get_xivo_gt_filename(args.dump, args.dataset, args.seq,
    sen=args.sen)

  ph = PlotHelper(args.seq, gt_data, estimator_data)

  # ground truth alignment
  ph.align_gt_to_est()

  # compute key quantities
  ph.compute_errors()

  # make plots
  ph.show_state_errors()
  ph.plot_trajectories()
  ph.plot_gauge_group()

  plt.show()
