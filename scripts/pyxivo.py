import argparse
import os, glob

import numpy as np
import glog
from transforms3d.quaternions import mat2quat

import sys
sys.path.insert(0, 'lib')
import pyxivo

TP_ROOT = '/home/feixh/Data/tumvi/exported/euroc/512_16'
KIF_ROOT = '/local2/Data/tumvi/exported/euroc/512_16'

parser = argparse.ArgumentParser()
parser.add_argument(
    '-root', default=KIF_ROOT, help='root directory of the tumvi dataset')
parser.add_argument(
    '-cfg', default='cfg/estimator.json', help='path to the estimator configuration')
parser.add_argument(
    '-seq', default='room6', help='short tag for the seuqence name')
parser.add_argument(
    '-cam_id', default=0, type=int, help='specify which camera to use')
parser.add_argument(
    '-out_dir', default='.', help='output directory to save results')
parser.add_argument(
    '-use_viewer', default=False, action='store_true',
    help='visualize trajectory and feature tracks if set')
args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    ########################################
    # LOAD DATA
    ########################################
    img_dir = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                           'mav0', 'cam{}'.format(args.cam_id), 'data')

    imu_path = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                            'mav0', 'imu0', 'data.csv')

    data = []

    for p in glob.glob(os.path.join(img_dir, '*.png')):
        ts = int(os.path.basename(p)[:-4])
        data.append((ts, p))

    with open(imu_path, 'r') as fid:
        for l in fid.readlines():
            if l[0] != '#':
                v = l.strip().split(',')
                ts = int(v[0])
                w = [float(x) for x in v[1:4]]
                t = [float(x) for x in v[4:]]
                data.append((ts, (w, t)))

    data.sort(key=lambda tup: tup[0])


    ########################################
    # INITIALIZE ESTIMATOR
    ########################################
    estimator = pyxivo.Estimator(args.cfg, 'cfg/viewer.json' if args.use_viewer else '', args.seq)
    results = []
    for i, (ts, content) in enumerate(data):
        # if i > 0 and i % 500 == 0:
        #     print('{:6}/{:6}'.format(i, len(data)))
        if isinstance(content, tuple):
            gyro, accel = content
            estimator.InertialMeas(ts, gyro[0], gyro[1], gyro[2], accel[0],
                                   accel[1], accel[2])
        else:
            estimator.VisualMeas(ts, content)
            estimator.Visualize()

            now = estimator.now()
            gsb = np.array(estimator.gsb())
            Tsb = gsb[:, 3]

            # print gsb[:3, :3]
            try:
                q = mat2quat(gsb[:3, :3])  # [w, x, y, z]
                # format compatible with tumvi rgbd benchmark scripts
                results.append(
                    [now * 1e-9, Tsb[0], Tsb[1], Tsb[2], q[1], q[2], q[3], q[0]])
            except np.linalg.linalg.LinAlgError:
                pass


    np.savetxt(
        os.path.join(args.out_dir, 'tumvi_cam{}_{}'.format(args.cam_id, args.seq)),
        results,
        fmt='%f %f %f %f %f %f %f %f')

    mocap_path = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                              'mav0', 'mocap0', 'data.csv')

    gt = []
    with open(mocap_path, 'r') as fid:
        for l in fid.readlines():
            if l[0] != '#':
                v = l.strip().split(',')
                ts = int(v[0])
                t = [float(x) for x in v[1:4]]
                q = [float(x) for x in v[4:]]  # [w, x, y, z]
                gt.append(
                    [ts * 1e-9, t[0], t[1], t[2], q[1], q[2], q[3], q[0]])

    np.savetxt(
        os.path.join(args.out_dir, 'tumvi_{}_gt'.format(args.seq)),
        gt,
        fmt='%f %f %f %f %f %f %f %f')
