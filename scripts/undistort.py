import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('log', help='log file dumped by pyxivo to process')
parser.add_argument('out_dir', help='output directory to save undistorted images')
parser.add_argument('-debug', action='store_true', default=False, help='if set, show images')


def createUndistortionLUT():
    fx = 190.97847715128717
    fy = 190.9733070521226
    cx = 254.93170605935475
    cy = 256.8974428996504

    K = np.array([[fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]])

    distortion = np.array([0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182])

    # Note: the desired size & projection matrix are not necessarilty
    # the same as the input one; pick what you like.
    desiredSize = (512, 512)
    desiredProj = K

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, None, desiredProj, desiredSize, cv2.CV_32FC1)
    return map1, map2


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    lut = createUndistortionLUT()
    with open(args.log, 'r') as fid:
        data = json.load(fid)
    try:
        for each in data:
            image = cv2.imread(each['ImagePath'])
            ts = each['Timestamp']
            Tsb = np.array(each['TranslationXYZ'])
            Qsb = np.array(each['QuaternionWXYZ'])

            undistorted = cv2.remap(image, lut[0], lut[1], cv2.INTER_LINEAR)
            outputPath = os.path.join(args.out_dir, '{}.png'.format(ts))
            cv2.imwrite(outputPath, undistorted)

            if args.debug:
                cv2.imshow('distorted image'.format(ts), image)
                cv2.imshow('undistorted image'.format(ts), undistorted)
                cv2.waitKey(30)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
