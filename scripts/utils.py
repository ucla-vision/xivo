import os

import numpy as np

def get_xivo_output_filename(dumpdir, dataset, seq, cam_id=0, sen='tango_top'):
  if dataset=="tumvi":
    estimator_datafile = os.path.join(dumpdir,
      "tumvi_{}_cam{}".format(seq, cam_id))
  elif dataset=="cosyvio":
    estimator_datafile = os.path.join(dumpdir,
      "cosyvio_{}_{}".format(sen, seq))
  else:
    estimator_datafile = os.path.join(dumpdir, "{}_{}".format(dataset, seq))
  return estimator_datafile


def get_xivo_gt_filename(dumpdir, dataset, seq, sen='tango_top'):
  if dataset=="cosyvio":
    gt_data = os.path.join(dumpdir, "cosyvio_{}_{}_gt".format(sen, seq))
  else:
    gt_data = os.path.join(dumpdir, "{}_{}_gt".format(dataset, seq))
  return gt_data


def upper_triangular_list(square_matrix, return_numpy=False, ret_dim=True):
    """Used on covariance matrices so we're not printing out as many digits. Save the
    upper-triangle of a square matrix as a list of numbers, row-major."""
    dim = np.shape(square_matrix)[0]
    l = []
    for i in range(dim):
        for j in range(i,dim):
            l.append(square_matrix[i][j])
    if return_numpy:
      l = np.array(l)
    if ret_dim:
      return [dim, l]
    else:
      return l
