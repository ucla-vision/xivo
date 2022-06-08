import re
import json
import os

import numpy as np

def get_xivo_output_filename(dumpdir, dataset, seq, cam_id=0, sen='tango_top'):
  if dataset in ["tumvi", "euroc"]:
    estimator_datafile = os.path.join(dumpdir,
      "{}_{}_cam{}".format(dataset, seq, cam_id))
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


def cleanup_and_load_json(results_filename):
  with open(results_filename, 'r') as fp:
    filestr = ""
    for line in fp:
      # delete all text after "//"
      if "//" in line:
        idx = line.index("//")
        line = line[:idx]
      sub1 = re.sub('nan', 'NaN', line)
      sub2 = re.sub('inf', 'Infinity', sub1)
      filestr = filestr + sub2
    return json.loads(filestr)


def state_indices(state_portion):
  if state_portion=="W":
    ind_l = 0
    ind_u = 3
  elif state_portion=="T":
    ind_l = 3
    ind_u = 6
  elif state_portion=="V":
    ind_l = 6
    ind_u = 9
  elif state_portion=="WT":
    ind_l = 0
    ind_u = 6
  elif state_portion=="WTV":
    ind_l = 0
    ind_u = 9
  else:
    raise ValueError("invalid state portion")
  return (ind_l, ind_u)


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


def from_upper_triangular_list(dim, l):
  """reverses function `upper_triangular_list` in file savers.py"""
  mat = np.zeros((dim,dim))
  ind = 0
  for i in range(dim):
    for j in range(i,dim):
      mat[i,j] = l[ind]
      mat[j,i] = mat[i,j]
      ind += 1
  return mat


def scale_covariance_matrix(cov, factor):
  U,S,Vh = np.linalg.svd(cov)
  scaled_S = S / factor
  return U.dot(np.diag(scaled_S).dot(Vh))


def rigid_transform_3d(A, B):
  # Input: expects Nx3 matrix of points
  # Returns R,t
  # R = 3x3 rotation matrix
  # t = 3x1 column vector
  assert len(A) == len(B)

  N = A.shape[0]  # total points
  centroid_A = np.mean(A, axis=0)
  centroid_B = np.mean(B, axis=0)
  # centre the points
  AA = A - np.tile(centroid_A, (N, 1))
  BB = B - np.tile(centroid_B, (N, 1))
  # dot is matrix multiplication for array
  H = np.matmul(np.transpose(AA), BB)
  U, _, Vt = np.linalg.svd(H)
  R = np.matmul(Vt.T, U.T)

  # special reflection case
  if np.linalg.det(R) < 0:
    Vt[2,:] *= -1
    R = np.matmul(Vt.T, U.T)

  t = np.matmul(-R, centroid_A.T) + centroid_B.T

  return R, t

