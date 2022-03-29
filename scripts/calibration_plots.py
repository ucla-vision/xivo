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


def camera_type_to_labels(camtype):
  if camtype == 0:  # pinhole
    return [ "fx", "fy", "cx", "cy" ]
  elif camtype == 1: # atan
    return [ "fx", "fy", "cx", "cy", "w" ]
  elif camtype == 2: # radtan
    return [ "fx", "fy", "cx", "cy", "p1", "p2", "k1", "k2", "k3" ]
  elif camtype == 3:
    return [ "fx", "fy", "cx", "cy", "k0", "k1", "k2", "k3" ]


class CalibPlotHelper:
  def __init__(self, seq, estimator_results, start_ind=0, end_ind=None):
    self.seq = seq

    # Text for x-axis time plots
    self.time_axis_label = 'Timesteps'

    # load estimator data
    self.est = EstimatorData(estimator_results, start_ind=start_ind,
      end_ind=end_ind)

    self.time_axis = np.linspace(0, self.est.nposes-1, self.est.nposes)


  def plot_calib_states(self):
    time_three_plots(self.time_axis, self.est.Wbc, "Online Wbc Calibration",
                    xlabel=self.time_axis_label)
    time_three_plots(self.time_axis, self.est.Tbc, "Online Tbc Calibration",
                     xlabel=self.time_axis_label)
    time_three_plots(self.time_axis, self.est.ba, "Online ba Calibration",
                     xlabel=self.time_axis_label)
    time_three_plots(self.time_axis, self.est.bg, "Online bg Calibration",
                     xlabel=self.time_axis_label)
    time_three_plots(self.time_axis, self.est.Wg, "Online Wg Calibration",
                     xlabel=self.time_axis_label)
    time_plot(self.time_axis, self.est.td, "Online td Calibration",
              xlabel=self.time_axis_label, ylabel="Offset (s)")
    matrix_frodiff_plot(self.time_axis, self.est.Ca, "Online Ca Calibration",
                        xlabel=self.time_axis_label)
    matrix_frodiff_plot(self.time_axis, self.est.Cg, "Online Cg Calibration",
                        xlabel=self.time_axis_label)

    camera_labels = camera_type_to_labels(self.est.camera_type)
    time_n_plots(self.time_axis, self.est.camera[:len(camera_labels),:],
                 "Online Camera Calibration", ylabels=camera_labels,
                 xlabel=self.time_axis_label)


if __name__ == "__main__":
  args = parser.parse_args()

  estimator_data = get_xivo_output_filename(args.dump, args.dataset, args.seq,
                                            cam_id=args.cam_id, sen=args.sen)

  ph = CalibPlotHelper(args.seq, estimator_data)

  ph.plot_calib_states()

  plt.show()
