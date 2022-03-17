import numpy as np
import matplotlib.pyplot as plt


def time_plot(time_axis, signal, title=None, ylabel=None, xlabel=None):
  plt.figure()
  _time_plot(time_axis, signal, title=title, ylabel=ylabel, xlabel=xlabel)


def _time_plot(time_axis, signal, title=None, ylabel=None, xlabel=None):
  plt.plot(time_axis, signal)
  if xlabel is not None:
    plt.xlabel(xlabel)
  if ylabel is not None:
    plt.ylabel(ylabel)
  if title is not None:
    plt.title(title)


def time_three_plots(time_axis, signals, suptitle, titles=None, xlabel=None):
  if titles is None:
    titles = [None, None, None]
  
  plt.figure()
  plt.suptitle(suptitle)
  plt.subplot(3,1,1)
  _time_plot(time_axis, signals[0,:], title=titles[0], ylabel="x-axis")
  plt.subplot(3,1,2)
  _time_plot(time_axis, signals[1,:], title=titles[1], ylabel="y-axis")
  plt.subplot(3,1,3)
  _time_plot(time_axis, signals[2,:], title=titles[2], ylabel="z-axis",
    xlabel=xlabel)


def matrix_frodiff_plot(time_axis, matrix_signal, title):
  first_mat = matrix_signal[:,:,0]
  diff_signal = np.zeros(len(time_axis))
  for i in range(len(time_axis)):
    diff_signal[i] = np.linalg.norm(matrix_signal[:,:,i] - first_mat)

  plt.figure()
  plt.title(title)
  plt.plot(time_axis, diff_signal)
  plt.xlabel("Time (s)")
  plt.ylabel("Frobenius Norm from Initial Value")


def error_three_plots(time_axis, error_signals, seq, error_type, error_unit):
  means = np.mean(error_signals, axis=1)
  var = np.var(error_signals, axis=1)
  print("{} {} error means: {} {}".format(seq, error_type, means, error_unit))
  print("{} {} error variance: {}".format(seq, error_type, var))

  plt.figure()
  plt.suptitle("{} {} error ({})".format(seq, error_type, error_unit))
  plt.subplot(3,1,1)
  _time_plot(time_axis, error_signals[0,:],
    "x-axis error mean/var: {0:10.3g}, {1:10.3g}".format(means[0], var[0]))
  plt.subplot(3,1,2)
  _time_plot(time_axis, error_signals[1,:],
    "y-axis error mean/var: {0:10.3g}, {1:10.3g}".format(means[1], var[1]))
  plt.subplot(3,1,3)
  _time_plot(time_axis, error_signals[2,:],
    "z-axis error mean/var: {0:10.3g}, {1:10.3g}".format(means[2], var[2]))


def plot_3D_error_cloud(error_signal, title):
  fig = plt.figure()
  plt.suptitle(title)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(error_signal[0,:], error_signal[1,:], error_signal[2,:],
    marker='.')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')


def plot_3d_trajectories(Tsb_est, Tsb_gt, units='m'):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(Tsb_est[0,:], Tsb_est[1,:], Tsb_est[2,:], c='r')
  ax.plot(Tsb_gt[0,:], Tsb_gt[1,:], Tsb_gt[2,:], c='b')
  ax.set_xlabel('x ({})'.format(units))
  ax.set_ylabel('y ({})'.format(units))
  ax.set_zlabel('z ({})'.format(units))
  ax.legend(('estimated', 'ground truth'))


def error_three_plots(time_axis, error_signals, seq, error_type, error_unit):
  means = np.mean(error_signals, axis=1)
  var = np.var(error_signals, axis=1)
  print("{} {} error means: {} {}".format(seq, error_type, means, error_unit))
  print("{} {} error variance: {}".format(seq, error_type, var))

  plt.figure()
  plt.suptitle("{} {} error ({})".format(seq, error_type, error_unit))
  plt.subplot(3,1,1)
  _time_plot(time_axis, error_signals[0,:],
    "x-axis error mean/var: {0:10.3g}, {1:10.3g}".format(means[0], var[0]))
  plt.subplot(3,1,2)
  _time_plot(time_axis, error_signals[1,:],
    "y-axis error mean/var: {0:10.3g}, {1:10.3g}".format(means[1], var[1]))
  plt.subplot(3,1,3)
  _time_plot(time_axis, error_signals[2,:],
    "z-axis error mean/var: {0:10.3g}, {1:10.3g}".format(means[2], var[2]))


def plot_3D_error_cloud(error_signal, title):
  fig = plt.figure()
  plt.suptitle(title)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(error_signal[0,:], error_signal[1,:], error_signal[2,:],
    marker='.')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')


def time_plot(time_axis, signal, title=None, ylabel=None, xlabel=None):
  plt.figure()
  _time_plot(time_axis, signal, title=title, ylabel=ylabel, xlabel=xlabel)


def _time_plot(time_axis, signal, title=None, ylabel=None, xlabel=None):
  plt.plot(time_axis, signal)
  if xlabel is not None:
    plt.xlabel(xlabel)
  if ylabel is not None:
    plt.ylabel(ylabel)
  if title is not None:
    plt.title(title)

