import sys, json, os
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import cv2

import pdb


FEATURE_TYPES = ["sift", "surf", "orb"]
WINDOW_SIZES = [5, 10, 15, 20]
DATAROOT = "/local2/Data/tumvi_undistorted"
EPIPOLAR_TOL = 0.025
USE_RANSAC = True
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
    def __init__(self, dataroot, room_number, feature_type):
        self.dataroot = dataroot
        self.room_number = room_number
        self.mask = None
    
        # Open up ground truth file and load data
        gt_file = open(dataroot + "/tumvi_room" + str(room_number) + "_cam0", "r")
        self.data = json.load(gt_file)

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

        # For storing pts and timestamps
        #self.stored_points = [ [] for i in range(len(self.data)) ]
        self.stored_points = {}
        self.timestamps = [ 0 for i in range(len(self.data)) ]

        # Place to dump text files full of points
        self.pointlist_dir = dataroot + "/featurelists/" + feature_type + \
            "/room" + str(self.room_number)
        if not os.path.exists(self.pointlist_dir):
            os.makedirs(self.pointlist_dir)


    def compute_projection_matrix(self, Q_wxyz, T_xyz):
        Q_xyzw = [Q_wxyz[1], Q_wxyz[2], Q_wxyz[3], Q_wxyz[0]]
        R = Rotation.from_quat(Q_xyzw)
        T = np.array(T_xyz).reshape((3,1))
        RT = np.hstack((R.as_dcm(), T))
        return self.K.dot(RT)


    def get_pose_and_filename(self, ind):
        timestamp = self.data[ind]["Timestamp"] # integer type
        TranslationXYZ = np.array(self.data[ind]["TranslationXYZ"]) # list of 3 floats
        QuaternionWXYZ = np.array(self.data[ind]["QuaternionWXYZ"]) # list of 4 floats
        undistort_img_filename = self.dataroot + "/undistorted/room" + \
            str(self.room_number) + "/" + str(timestamp) + ".png"
        return (timestamp, TranslationXYZ, QuaternionWXYZ, undistort_img_filename)


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


    def outlier_rejection(self, Q0_wxyz, Q1_wxyz, T0_xyz, T1_xyz, matches, pts0, pts1,
        tol):

        # find transformation that transforms q0 to q1
        Q0_xyzw = [Q0_wxyz[1], Q0_wxyz[2], Q0_wxyz[3], Q0_wxyz[0]]
        Q1_xyzw = [Q1_wxyz[1], Q1_wxyz[2], Q1_wxyz[3], Q1_wxyz[0]]
        R_s_0 = Rotation.from_quat(Q0_xyzw)
        R_s_1 = Rotation.from_quat(Q1_xyzw)
        T_s_0 = T0_xyz
        T_s_1 = T1_xyz

        R_1_s = R_s_1.inv()
        R_1_0 = R_1_s * R_s_0
        T_1_0 = R_1_s.apply(T_s_0 - T_s_1)
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


    def triangulate_pair(self, idx0, idx1, plot_matches, use_RANSAC, epi_tol,
                         debug=False):
        # load two images
        (timestamp0, T0_xyz, Q0_wxyz, file0) = self.get_pose_and_filename(idx0)
        (timestamp1, T1_xyz, Q1_wxyz, file1) = self.get_pose_and_filename(idx1)
        img0 = cv2.imread(file0)
        img1 = cv2.imread(file1)
        self.timestamps[idx0] = timestamp0
        self.timestamps[idx1] = timestamp1

        # Get 3 x 4 projection matrices for two instants in time
        Proj0 = self.compute_projection_matrix(Q0_wxyz, T0_xyz)
        Proj1 = self.compute_projection_matrix(Q1_wxyz, T1_xyz)

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
        if use_RANSAC:
            (_, good_match_list) = cv2.findHomography(np.transpose(points0),
                np.transpose(points1), method=cv2.RANSAC,
                ransacReprojThreshold=3, maxIters=2000, confidence=0.995)
            good_match_list = np.resize(good_match_list, n_matches)
            matches = prune_matches(matches, good_match_list)
        else:
            good_match_list = self.outlier_rejection(Q0_wxyz, Q1_wxyz, T0_xyz, T1_xyz,
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
        for i in range(n_good_matches):
            self.add_4d_point(timestamp0, points0[:,i], points4d[:,i])
            self.add_4d_point(timestamp1, points1[:,i], points4d[:,i])
        
        # Print results
        print("Frame {}/{}: {}/{} features, {} matches, {} good matches".format(
            idx0, idx1, len(kp0), len(kp1), n_matches, n_good_matches
        ))



    def triangulate_all_pairs(self, timestep_window_size, use_RANSAC=False,
        epi_tol=1e-3, plot_matches=False):
        num_pairs = len(self.data) - timestep_window_size

        for i in range(num_pairs):
            self.triangulate_pair(i,i+timestep_window_size, use_RANSAC=use_RANSAC,
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
    for feature_type in FEATURE_TYPES:
        print("FEATURE TYPE: {}".format(feature_type))
        for window_size in WINDOW_SIZES:
            mdc = MonoGTDepthCalc(DATAROOT, room, feature_type)
            mdc.triangulate_all_pairs(window_size, use_RANSAC=USE_RANSAC,
                                    epi_tol=EPIPOLAR_TOL, plot_matches=PLOT_MATCHES)
            mdc.write_all_files()

