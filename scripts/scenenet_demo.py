import argparse
import os, glob
import pickle
import matplotlib.pyplot as plt

import numpy as np
import glog
from transforms3d.quaternions import mat2quat
from transforms3d.axangles import mat2axangle
import cv2

import sys
sys.path.insert(0, 'lib')
import pyxivo

TP_ROOT = '/home/feixh/Data/tumvi/exported/euroc/512_16'
KIF_ROOT = '/local2/Data/tumvi/exported/euroc/512_16'

parser = argparse.ArgumentParser()
parser.add_argument(
    '-data', default='xivo_data_0_223.pkl', help='data to play')
parser.add_argument(
    '-use_viewer', default=False, action='store_true',
    help='visualize trajectory and feature tracks if set')
args = parser.parse_args()

def SO3log(R):
    # print R
    ax, a = mat2axangle(R)
    return ax * a

def compareStates(est, gt):
    est = np.array(est) if est is not None else None
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.title('T' + chr(ord('x') + i))
        if est is not None:
            plt.plot(est[:, 0], est[:, i+1], 'b', label='estimation')
        if gt is not None:
            plt.plot([tup[0] for tup in gt],
                    [tup[1][i, 3] for tup in gt], 'r', label='ground truth')
        plt.legend(loc='upper left')

    for i in range(3):
        plt.subplot(2, 3, i+4)
        plt.title('W' + chr(ord('x') + i))
        if est is not None:
            plt.plot(est[:, 0], est[:, i+4], 'b', label='estimation')
        if gt is not None:
            plt.plot([tup[0] for tup in gt],
                    [SO3log(tup[1][:3, :3])[i] for tup in gt], 'r', label='ground truth')
        plt.legend(loc='upper left')

    plt.show()

if __name__ == '__main__':
    with open(args.data, 'rb') as fid:
        data = pickle.load(fid)
    data, mocap_data = data['image_data'] + data['imu_data'], data['mocap_data']
    data.sort(key=lambda tup: tup[0])

    ########################################
    # INITIALIZE ESTIMATOR
    ########################################
    estimator = pyxivo.Estimator('cfg/estimator_scenenet.json',
            'cfg/viewer_scenenet.json' if args.use_viewer else '',
            'Demo-SceneNet')
    results = []
    results_w = []
    # create a temp folder to hold debug info
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    for i, item in enumerate(data):
        if i > 0 and i % 500 == 0:
            print('{:6}/{:6}'.format(i, len(data)))
        ts = int(item[0] * 1e9)
        if len(item) == 3:
            gyro, accel = item[1:]
            estimator.InertialMeas(ts,
                    gyro[0], gyro[1], gyro[2],
                    accel[0], accel[1], accel[2])
        else:
            image = item[1]
            cv2.imwrite('tmp/{:04}.jpg'.format(i), image) # super hack: dump image and read ...
            estimator.VisualMeas(ts, 'tmp/{:04}.jpg'.format(i))
            estimator.Visualize()

        gsb = np.array(estimator.gsb())
        Tsb = gsb[:, 3]

        q = mat2quat(gsb[:3, :3])  # [w, x, y, z]
        # format compatible with tumvi rgbd benchmark scripts
        results.append(
            [ts * 1e-9, Tsb[0], Tsb[1], Tsb[2], q[1], q[2], q[3], q[0]])

        w = SO3log(gsb[:3, :3])
        results_w.append(
            [ts * 1e-9, Tsb[0], Tsb[1], Tsb[2], w[0], w[1], w[2]])

    compareStates(results_w, mocap_data)

    np.savetxt(
        os.path.join('scenenet_estimate'),
        results,
        fmt='%f %f %f %f %f %f %f %f')

    gt = []
    for ts, gwc in mocap_data:
        T = gwc[:3, 3]
        q = mat2quat(gwc[:3, :3])
        gt.append([ts, T[0], T[1], T[2], q[1], q[2], q[3], q[0]])

    np.savetxt(
        os.path.join('scenenet_gt'),
        gt,
        fmt='%f %f %f %f %f %f %f %f')
