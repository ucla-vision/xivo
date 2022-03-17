import sys, os
import json
import numpy as np
from scipy.spatial.transform import Rotation

from utils import (from_upper_triangular_list, scale_covariance_matrix,
                   cleanup_and_load_json)
from to_json import to_json


def Q_wxyz_to_Rotation(json_item):
  Q_wxyz = np.array(json_item)
  Q_xyzw = np.array([Q_wxyz[1], Q_wxyz[2], Q_wxyz[3], Q_wxyz[0]])
  R = Rotation.from_quat(Q_xyzw)
  return R


class EstimatorData:
  def __init__(self, results_file, start_ind=0, end_ind=None):

    self.start_ind = start_ind
    self.end_ind = end_ind

    # open file to do nan/NaN and inf/Infinity replacements
    self.estimator_results = cleanup_and_load_json(results_file)

    if self.end_ind is None:
      self.nposes = len(self.estimator_results["data"]) - self.start_ind
      self.end_ind = self.nposes
    else:
      self.nposes = self.end_ind - self.start_ind

    (total_dim,_) = self.estimator_results["data"][0]["Pstate"]
    self.time_axis = np.zeros((self.nposes,1))
    self.Rsb = []
    self.Tsb = np.zeros((3,self.nposes))
    self.Vsb = np.zeros((3,self.nposes))
    self.P = np.zeros((total_dim, total_dim, self.nposes))

    self.MeasurementUpdateInitialized = []
    self.inn_Wsb = np.zeros((3,self.nposes))
    self.inn_Tsb = np.zeros((3,self.nposes))
    self.inn_Vsb = np.zeros((3,self.nposes))

    # lists of lists for the map.
    self.feature_ids = []
    self.feature_positions = []
    self.feature_covariances = []
    self.group_ids = []
    self.group_Tsb = []
    self.group_Rsb = []
    self.group_covariances = []

    self.gauge_group = []

    # calibration states
    self.Wbc = np.zeros((3,self.nposes))
    self.Tbc = np.zeros((3,self.nposes))
    self.ba = np.zeros((3,self.nposes))
    self.bg = np.zeros((3,self.nposes))
    self.Wg = np.zeros((3,self.nposes))
    self.td = np.zeros(self.nposes)
    self.Ca = np.zeros((3,3,self.nposes))
    self.Cg = np.zeros((3,3,self.nposes))
    self.camera = np.zeros((9,self.nposes))
    self.camera_type = 0

    # sample covariance data
    self.has_sample_cov = []
    self.sample_covWTV = np.zeros((9,9,self.nposes))

    # collect the pose data
    self.collect_data()


  def scale_covariance_matries(self, factor):
    for i in range(self.nposes):
      self.P[:,:,i] = scale_covariance_matrix(self.P[:,:,i], factor)


  def collect_data(self):
    for i in range(self.nposes):
      timestamp, R_sb, T_sb, V_sb, P, inn_Wsb, inn_Tsb, inn_Vsb, gauge_group, \
        W_bc, T_bc, ba, bg, Wg, td, Ca, Cg, Cam, CamType = \
          self.get_estimator_point(self.start_ind+i)
      self.Rsb.append(R_sb)
      self.Tsb[:,i] = T_sb
      self.Vsb[:,i] = V_sb
      self.P[:,:,i] = P
      self.inn_Wsb[:,i] = inn_Wsb
      self.inn_Tsb[:,i] = inn_Tsb
      self.inn_Vsb[:,i] = inn_Vsb
      self.time_axis[i] = timestamp
      self.gauge_group.append(gauge_group)

      feature_ids, feature_positions, feature_covs = \
          self.collect_feature_data(self.start_ind+i)
      self.feature_ids.append(feature_ids)
      self.feature_positions.append(feature_positions)
      self.feature_covariances.append(feature_covs)

      group_ids, group_Rsb, group_Tsb, group_covs = \
          self.collect_group_data(self.start_ind+i)
      self.group_ids.append(group_ids)
      self.group_Rsb.append(group_Rsb)
      self.group_Tsb.append(group_Tsb)
      self.group_covariances.append(group_covs)

      self.Wbc[:,i] = W_bc
      self.Tbc[:,i] = T_bc
      self.ba[:,i] = ba
      self.bg[:,i] = bg
      self.Wg[:,i] = Wg
      self.td[i] = td
      self.Ca[:,:,i] = Ca
      self.Cg[:,:,i] = Cg
      self.camera[:,i] = Cam
      if i==1:
        self.camera_type = CamType


      # read sample covariance
      cov = self.get_sample_cov(self.start_ind+i)
      if cov is not None:
        self.has_sample_cov.append(True)
        self.sample_covWTV[:,:,i] = cov
      else:
        self.has_sample_cov.append(False)


  def get_sample_cov(self, ind):
    data = self.estimator_results["data"][ind]
    if "has_sample_cov" in data:
      if data["has_sample_cov"]:
        (cov_dimWTV, sample_covWTV) = data["sample_cov_WTV"]
        return from_upper_triangular_list(cov_dimWTV, sample_covWTV)
      else:
        return None


  def get_estimator_point(self, ind):
    data = self.estimator_results["data"][ind]

    timestamp = data["Timestamp"]

    # Rotation, translation, velocity
    T_sb = np.array(data["Tsb_XYZ"])
    V_sb = np.array(data["Vsb_XYZ"])
    R_sb = Q_wxyz_to_Rotation(data["qsb_WXYZ"])

    # Covariance
    (Pdim, Plist) = data["Pstate"]
    P = from_upper_triangular_list(Pdim, Plist)

    # innovation
    inn_Wsb = np.array(data["inn_Wsb"])
    inn_Tsb = np.array(data["inn_Tsb"])
    inn_Vsb = np.array(data["inn_Vsb"])

    # gauge group
    group = data["group"]

    # Calibration states
    W_bc = Q_wxyz_to_Rotation(data["qbc_WXYZ"]).as_rotvec()
    T_bc = np.array(data["Tbc_XYZ"])
    W_g = Q_wxyz_to_Rotation(data["qg_WXYZ"]).as_rotvec()
    bg = np.array(data["bg"])
    ba = np.array(data["ba"])

    # optional calibration states
    td = float(data["td"])
    Ca = from_upper_triangular_list(3, data["Ca"])
    Cg = from_upper_triangular_list(3, data["Cg"])
    Cam = np.array(data["camera_intrinsics"])
    CamType = int(data["camera_type"])

    return (timestamp, R_sb, T_sb, V_sb, P, inn_Wsb, inn_Tsb, inn_Vsb, group,
            W_bc, T_bc, ba, bg, W_g, td, Ca, Cg, Cam, CamType)


  def collect_feature_data(self, ind):
    data = self.estimator_results["data"][ind]

    feature_ids = np.array(data["feature_ids"])
    feature_covs = []
    feature_positions = []

    for idx,_ in enumerate(feature_ids):
      cov_data = from_upper_triangular_list(3,
        data["feature_covs"][6*idx:6*(idx+1)])
      feature_pos = np.array(data["feature_positions"][3*idx:3*(idx+1)])

      feature_covs.append(cov_data)
      feature_positions.append(feature_pos)

    return (feature_ids, feature_positions, feature_covs)


  def collect_group_data(self, ind):
    data = self.estimator_results["data"][ind]

    group_ids = np.array(data["group_ids"])
    group_covs = []
    group_Rsb = []
    group_Tsb = []

    for idx,_ in enumerate(group_ids):
      cov_data = from_upper_triangular_list(6,
        data["group_covs"][21*idx:21*(idx+1)])
      pose_data = np.array(data["group_poses"][7*idx:7*(idx+1)])
      Qsb_xyzw = pose_data[0:4]
      Tsb = pose_data[4:]
      Rsb = Rotation.from_quat(Qsb_xyzw)

      group_covs.append(cov_data)
      group_Rsb.append(Rsb)
      group_Tsb.append(Tsb)
    
    return (group_ids, group_Rsb, group_Tsb, group_covs)


  def assign_val(self, ind, fieldname, value):
    self.estimator_results["data"][self.start_ind+ind][fieldname] = value


  def add_param(self, fieldname, value):
    self.estimator_results[fieldname] = value


  def write_json(self, filename):
    json_string = to_json(self.estimator_results)
    with open(filename, 'w') as fid:
      fid.write(json_string)


