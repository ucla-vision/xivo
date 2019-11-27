import argparse, json, os
import numpy as np
from scipy.spatial.transform import Rotation
import cv2


DATAROOT = "/local2/Data/tumvi_undistorted"
TIMESTEP_WINDOW_SIZE = 25

class Point:
    def __init__(self, pt2d, pt4d):
        self.pt = [int(pt2d[0]), int(pt2d[1])]
        self.pt4d = pt4d
    def __eq__(self, other):
        return (self.pt == other.pt)



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
        TranslationXYZ = self.data[ind]["TranslationXYZ"] # list of 3 floats
        QuaternionWXYZ = self.data[ind]["QuaternionWXYZ"] # list of 4 floats
        undistort_img_filename = self.dataroot + "/undistorted/room" + \
            str(self.room_number) + "/" + str(timestamp) + ".png"
        return (timestamp, TranslationXYZ, QuaternionWXYZ, undistort_img_filename)


    def write_file(self, timestamp):
        file_name = self.pointlist_dir + "/" + str(timestamp) + ".csv"
        fp = open(file_name, "w")
        for point in self.stored_points[timestamp]:
            fp.write("{},{},{},{},{},{}\n".format(
                point.pt[0], point.pt[1],
                point.pt4d[0], point.pt4d[1], point.pt4d[2], point.pt4d[3]
            ))
        fp.close()


    def write_all_files(self):
        for timestamp in self.stored_points:
            self.write_file(timestamp)


    def triangulate_pair(self, idx0, idx1):
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
        points0 = np.zeros((2, n_matches))
        points1 = np.zeros((2, n_matches))
        for i,match in enumerate(matches):
            points0[:,i] = kp0[match.queryIdx].pt
            points1[:,i] = kp1[match.trainIdx].pt

        # Triangulate matches
        try:
            points4d = cv2.triangulatePoints(Proj0, Proj1, points0, points1)
        except:
            print("Triangulation Failed on frames {}/{}".format(idx0, idx1))
            return

        # Add points0 to file of last saved points
        if timestamp0 not in self.stored_points:
            self.stored_points[timestamp0] = []
        for i in range(n_matches):
            pt_obj = Point(points0[:,i], points4d[:,i])
            if pt_obj not in self.stored_points[timestamp0]:
                self.stored_points[timestamp0].append(pt_obj)

        # Save pts1
        if timestamp1 not in self.stored_points:
            self.stored_points[timestamp1] = []
        for i in range(n_matches):
            pt_obj = Point(points1[:,i], points4d[:,i])
            # need extra check because matcher sometimes turns up duplicates
            if pt_obj not in self.stored_points[timestamp1]:
                self.stored_points[timestamp1].append(pt_obj)




    def triangulate_all_pairs(self, timestep_window_size):
        num_pairs = len(self.data) - timestep_window_size

        for i in range(num_pairs):
            self.triangulate_pair(i,i+timestep_window_size)





if __name__ == "__main__":
    for feature_type in ["sift", "surf", "orb"]:
        print("FEATURE TYPE: {}".format(feature_type))
        for i in range(1,6):
            print("ROOM {}".format(i))
            mdc = MonoGTDepthCalc(DATAROOT, i, feature_type)
            mdc.triangulate_all_pairs(TIMESTEP_WINDOW_SIZE)
            mdc.write_all_files()

