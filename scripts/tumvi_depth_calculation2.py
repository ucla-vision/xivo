import sys, json, os, argparse
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy import interpolate
import matplotlib.pyplot as plt
import cv2

import pdb

"""
- This script is the same as tumvi_depth_calculation.py (see notes at top of
  that file), except that it uses the ground truth value of gsb with the
  offline-calibrated gbc in order to get a more accurate value of gsc. With
  this more accurate value, we can use epipolar constraints instead of RANSAC
  for outlier rejection.

- In order to get the ground truth values, first run
      python scripts/run_and_eval_pyxivo.py -seq room[#]
  with all six indoor sequences so that the ground truth files are dumped into
  XIVO's root directory. (This file was written in a hurry.)

- The ground truth values for gsb are interpolated using Slerp and interp1d.
"""

FEATURE_TYPES = ["sift", "surf", "orb"]
WINDOW_SIZES = [5, 10, 15, 20]
DATAROOT = "/local2/Data/tumvi_undistorted"
EPIPOLAR_TOL = 5e-3
#USE_RANSAC = False
PLOT_MATCHES = False


def make_not_homogeneous(arr):
    arr  = arr / arr[-1]
    return arr[:-1]

def make_homogeneous(arr):
    return np.hstack((arr, np.array(1.0)))

def skew(x):
    return np.array([[    0, -x[2],  x[1]],
                     [ x[2],     0, -x[0]],
                     [-x[1],  x[0],     0]])


def prune_matches(match_list, good_match_list):
    new_match_list = []
    for ind,val in enumerate(good_match_list):
        if val==1:
            new_match_list.append(match_list[ind])
    return new_match_list


def interpolate_gsb(gt_timestamps, R_sb_list, T_sb_list, query_timestamp):
    # Interpolate translation
    gt_timestamps = np.array(gt_timestamps, dtype=np.float)
    T_sb_list = np.array(T_sb_list)
    interp_T = interpolate.interp1d(gt_timestamps, T_sb_list, axis=0)
    T_sb = interp_T(np.float(query_timestamp))

    # Interpolate Rotation (slerp)
    slerp = Slerp(gt_timestamps, R_sb_list)
    R_sb = slerp(np.array([query_timestamp]))

    return (R_sb, T_sb)
    


class Point:
    def __init__(self, pt2d, pt4d):
        self.pt = pt2d
        self.pt3d = make_not_homogeneous(pt4d)
        self.count = 1
    def __eq__(self, other):
        dist = np.linalg.norm(self.pt - other.pt)
        return (dist < 1)
    def add_pt3d(self, other_pt):
        new_count = self.count + 1
        self.pt3d = (1/new_count)*other_pt.pt3d + (self.count/new_count)*self.pt3d
        self.pt = (1/new_count)*other_pt.pt + (self.count/new_count)*self.pt
        self.count = new_count



class MonoGTDepthCalc:
    def __init__(self, dataroot, room_number, feature_type, use_ransac):
        self.dataroot = dataroot
        self.room_number = room_number
        self.mask = None
        self.use_ransac = use_ransac
    
        # Open up image data file and load metadata
        image_data_file = open(dataroot + "/tumvi_room" + str(room_number) + "_cam0", "r")
        self.image_data = json.load(image_data_file)

        # Open up ground truth file and load timestamps and ground truth gsb
        gt_filename = "tumvi_room" + str(room_number) + "_gt"
        self.load_gt_file(gt_filename)

        # Get feature detector and matcher
        if feature_type == "sift":
            self.detector = cv2.xfeatures2d.SIFT_create()
            self.matcher = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=True)
        elif feature_type == "surf":
            self.detector = cv2.xfeatures2d.SURF_create()
            self.matcher = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=True)
        elif feature_type == "orb":
            self.detector = cv2.ORB_create()
            self.matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)

        # Camera projection matrix for six indoor rooms
        fx = 190.97847715128717
        fy = 190.9733070521226
        cx = 254.93170605935475
        cy = 256.8974428996504
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        self.K_inv = np.array([[1/fx,    0, -cx/fx],
                               [   0, 1/fy, -cy/fy],
                               [   0,    0,      1]])
        
        # g_bc from TUMVI, cam0
        self.T_bc = np.array([0.04557484, -0.0711618, -0.04468125])
        R_bc_dcm = np.array([
             [-0.99952504,  0.00750192, -0.02989013],
             [0.02961534, -0.03439736, -0.99896935],
             [-0.00852233, -0.99938008,  0.03415885]])
        self.R_bc = Rotation.from_dcm(R_bc_dcm)

        # For storing pts and timestamps
        self.stored_points = {}
        self.timestamps = [ 0 for i in range(len(self.image_data)) ]

        # Place to dump text files full of points
        if self.use_ransac:
            self.pointlist_dir = dataroot + "/featurelists_gt_ransac/" + feature_type + \
                "/room" + str(self.room_number)
        else:
            self.pointlist_dir = dataroot + "/featurelists_gt/" + feature_type + \
                "/room" + str(self.room_number)
        if not os.path.exists(self.pointlist_dir):
            os.makedirs(self.pointlist_dir)


    def load_gt_file(self, gt_filename):
        gt_file = open(gt_filename, "r")

        self.gt_timestamps = []
        self.gt_poses = {}

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
            self.gt_timestamps.append(timestamp_ns)
            self.gt_poses[timestamp_ns] = (R_sb, T_sb)

        self.gt_timestamps = np.array(self.gt_timestamps)


    def get_gt_gsc(self, timestamp):
        idx = np.abs(self.gt_timestamps - timestamp).argmin()
        closest_gt_timestamp = self.gt_timestamps[idx]
        (R_sb_0, T_sb_0) = self.gt_poses[closest_gt_timestamp]

        if (timestamp == closest_gt_timestamp) or (idx==0) or (idx==len(self.gt_timestamps)-1):
            R_sb = R_sb_0
            T_sb = T_sb_0
        elif timestamp > closest_gt_timestamp:
            other_gt_timestamp = self.gt_timestamps[idx+1]
            (R_sb_1, T_sb_1) = self.gt_poses[other_gt_timestamp]
            (R_sb, T_sb) = interpolate_gsb([closest_gt_timestamp, other_gt_timestamp],
                                           Rotation.from_quat([
                                               R_sb_0.as_quat(),
                                               R_sb_1.as_quat()
                                           ]),
                                           [T_sb_0, T_sb_1],
                                           timestamp)
        else:
            other_gt_timestamp = self.gt_timestamps[idx-1]
            (R_sb_1, T_sb_1) = self.gt_poses[other_gt_timestamp]
            (R_sb, T_sb) = interpolate_gsb([other_gt_timestamp, closest_gt_timestamp],
                                           Rotation.from_quat([
                                               R_sb_1.as_quat(),
                                               R_sb_0.as_quat()
                                           ]),
                                           [T_sb_1, T_sb_0],
                                           timestamp)
        R_sc = R_sb * self.R_bc
        T_sc = R_sb.apply(self.T_bc) + T_sb
        return (R_sc, T_sc)


    def compute_projection_matrix(self, R_sc, T_sc):
        R_cs = R_sc.inv()
        R_cs_dcm = np.reshape(R_cs.as_dcm(), (3,3))
        T_cs = -R_cs.apply(T_sc)
        T_cs = np.reshape(T_cs, (3,1))

        RT = np.hstack((R_cs_dcm, T_cs))
        return self.K.dot(RT)


    def get_pose_and_filename(self, ind):
        timestamp = self.image_data[ind]["Timestamp"] # integer type
        T_sc = np.array(self.image_data[ind]["TranslationXYZ"]) # list of 3 floats
        Q_sc_wxyz = np.array(self.image_data[ind]["QuaternionWXYZ"]) # list of 4 floats
        Q_sc_xyzw = np.array([Q_sc_wxyz[1], Q_sc_wxyz[2], Q_sc_wxyz[3], Q_sc_wxyz[0]])
        R_sc = Rotation.from_quat(Q_sc_xyzw)

        undistort_img_filename = self.dataroot + "/undistorted/room" + \
            str(self.room_number) + "/" + str(timestamp) + ".png"

        return (timestamp, T_sc, R_sc, undistort_img_filename)


    def write_file(self, timestamp):
        file_name = self.pointlist_dir + "/" + str(timestamp) + ".csv"
        fp = open(file_name, "w")
        for point in self.stored_points[timestamp]:
            fp.write("{},{},{},{},{}\n".format(
                point.pt[0], point.pt[1],
                point.pt3d[0], point.pt3d[1], point.pt3d[2]
            ))
        fp.close()


    def write_all_files(self):
        for timestamp in self.stored_points:
            self.write_file(timestamp)


    def outlier_rejection(self, R0_sc, R1_sc, T0_sc, T1_sc, matches, pts0, pts1,
        tol):

        # find transformation that transforms q0 to q1
        R_s_0 = R0_sc
        R_s_1 = R1_sc
        T_s_0 = T0_sc
        T_s_1 = T1_sc

        R_1_s = R_s_1.inv()
        R_1_0 = R_1_s * R_s_0
        T_1_0 = np.reshape(R_1_s.apply(T_s_0 - T_s_1), 3)
        E = skew(T_1_0).dot(R_1_0.as_dcm())

        # Go through matches and mark bad ones using the essential matrix
        good_match_list = np.ones(len(matches), dtype=np.uint8)
        for i in range(len(matches)):
            pt0_px = make_homogeneous(pts0[:,i])
            pt1_px = make_homogeneous(pts1[:,i])

            pt0_cam0 = self.K_inv.dot(pt0_px)
            pt1_cam1 = self.K_inv.dot(pt1_px)

            res = pt1_cam1.dot(E.dot(pt0_cam0))
            if abs(res) > tol:
                good_match_list[i] = 0

        return good_match_list


    def triangulate_pair(self, idx0, idx1, plot_matches, epi_tol, debug=False):
        # load two images
        (timestamp0, _, _, file0) = self.get_pose_and_filename(idx0)
        (timestamp1, _, _, file1) = self.get_pose_and_filename(idx1)
        img0 = cv2.imread(file0)
        img1 = cv2.imread(file1)
        self.timestamps[idx0] = timestamp0
        self.timestamps[idx1] = timestamp1

        # Get corresponding ground truth
        (R0_sc, T0_sc) = self.get_gt_gsc(timestamp0)
        (R1_sc, T1_sc) = self.get_gt_gsc(timestamp1)

        # Get 3 x 4 projection matrices for two instants in time
        Proj0 = self.compute_projection_matrix(R0_sc, T0_sc)
        Proj1 = self.compute_projection_matrix(R1_sc, T1_sc)

        # detect features and set up dictionary from descriptors to keypoints
        kp0, des0 = self.detector.detectAndCompute(img0, self.mask)
        kp1, des1 = self.detector.detectAndCompute(img1, self.mask)

        # Match features (des0 = query, des1 = database), we are trying to find
        # descriptors in img0 that correspond to descriptors in img1.
        try:
            matches = self.matcher.match(des0, des1)
        except:
            print("Matcher Failed on frames {}/{}".format(idx0, idx1))
            return
        n_matches = len(matches)
        if n_matches == 0:
            print("No matches on frames {}/{}".format(idx0, idx1))
            return
        points0 = np.zeros((2, n_matches))
        points1 = np.zeros((2, n_matches))
        for i,match in enumerate(matches):
            points0[:,i] = kp0[match.queryIdx].pt
            points1[:,i] = kp1[match.trainIdx].pt

        # plot matches before outlier rejection
        if plot_matches:
            img2 = cv2.drawMatches(img0, kp0, img1, kp1, matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img2)
            plt.show()

        # Outlier rejection
        if self.use_ransac:
            (_, good_match_list) = cv2.findEssentialMat(np.transpose(points0),
                np.transpose(points1), self.K, method=cv2.RANSAC, prob=0.999,
                threshold=1.0)
            good_match_list = np.resize(good_match_list, n_matches)
            matches = prune_matches(matches, good_match_list)
        else:
            good_match_list = self.outlier_rejection(R0_sc, R1_sc, T0_sc, T1_sc,
                matches, points0, points1, epi_tol)
            matches = prune_matches(matches, good_match_list)
        
        
        # plot matches after outlier rejection
        if plot_matches:
            img3 = cv2.drawMatches(img0, kp0, img1, kp1, matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3)
            plt.show()

        n_good_matches = np.count_nonzero(good_match_list)
        if n_good_matches==0:
            print("No good matches on frames {}/{}".format(idx0, idx1))
            return
        points0 = points0[:,good_match_list==1]
        points1 = points1[:,good_match_list==1]

        if debug:
            pdb.set_trace()

        # Triangulate matches
        try:
            points4d = cv2.triangulatePoints(Proj0, Proj1, points0, points1)
        except:
            print("Triangulation Failed on frames {}/{}".format(idx0, idx1))
            return

        # Add matches to lists of points for both images
        R0_cs = R0_sc.inv()
        R1_cs = R1_sc.inv()
        for i in range(n_good_matches):
            # Get homogeneous coordinate in world frame
            X_s = make_not_homogeneous(points4d[:,i])

            # Transform to camera frame
            X_c0 = R0_cs.apply(X_s - T0_sc)
            X_c1 = R1_cs.apply(X_s - T1_sc)
            X_c0 = np.reshape(X_c0, (3,))
            X_c1 = np.reshape(X_c1, (3,))
            self.add_4d_point(timestamp0, points0[:,i], make_homogeneous(X_c0))
            self.add_4d_point(timestamp1, points1[:,i], make_homogeneous(X_c1))
        
        # Print results
        print("Frame {}/{}: {}/{} features, {} matches, {} good matches".format(
            idx0, idx1, len(kp0), len(kp1), n_matches, n_good_matches
        ))



    def triangulate_all_pairs(self, timestep_window_size,
        epi_tol=1e-3, plot_matches=False):
        num_pairs = len(self.image_data) - timestep_window_size

        for i in range(num_pairs):
            self.triangulate_pair(i,i+timestep_window_size,
                                  epi_tol=epi_tol, plot_matches=plot_matches)


    def add_4d_point(self, timestamp, ptpx, pt4d):
        pt_obj = Point(ptpx, pt4d)

        if timestamp not in self.stored_points:
            self.stored_points[timestamp] = [pt_obj]
            return

        if pt_obj in self.stored_points[timestamp]:
            ind = self.stored_points[timestamp].index(pt_obj)
            self.stored_points[timestamp][ind].add_pt3d(pt_obj)
        else:
            self.stored_points[timestamp].append(pt_obj)



if __name__ == "__main__":
    room = sys.argv[1]
    USE_RANSAC = bool(int(sys.argv[2]))
    #mdc = MonoGTDepthCalc(DATAROOT, 1, "sift", USE_RANSAC)
    #mdc.triangulate_pair(1134, 1144, PLOT_MATCHES, EPIPOLAR_TOL)
    for feature_type in FEATURE_TYPES:
        print("FEATURE TYPE: {}".format(feature_type))
        for window_size in WINDOW_SIZES:
            mdc = MonoGTDepthCalc(DATAROOT, room, feature_type, USE_RANSAC)
            mdc.triangulate_all_pairs(window_size,
                                    epi_tol=EPIPOLAR_TOL, plot_matches=PLOT_MATCHES)
            mdc.write_all_files()
