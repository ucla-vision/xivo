import numpy as np
import argparse
import os, sys
from shutil import copyfile
from transforms3d.quaternions import quat2mat, mat2quat

sys.path.insert(0, 'scripts/tum_rgbd_benchmark_tools')
from evaluate_ate import align

TP_ROOT = '/home/feixh/Data/tumvi/exported/euroc/512_16'
KIF_ROOT = '/local2/Data/tumvi/exported/euroc/512_16'

parser = argparse.ArgumentParser()
parser.add_argument(
        '-root', default=KIF_ROOT, help='root directory of the tumvi dataset')
parser.add_argument(
        '-working-dir', required=True, help='working directory which holds the state estimates')
parser.add_argument(
        '-seq', required=True, help='which sequence to work on')
args = parser.parse_args()


if __name__ == '__main__':
    # load two trajectories
    trajs = []
    for cam_id in [0, 1]:
        data = np.loadtxt(os.path.join(args.working_dir, 'tumvi_{}_cam{}'.format(args.seq, args.cam_id)))
        # convert each data line to (timestamp, pose) tuple
        # layout of each line
        # 0: timestamp; 1-3: Tsb; 4-7: Rsb in quaternion [x, y, z, w] convention
        traj = []
        for line in data:
            ts = line[0]
            Tsb = line[1:4, np.newaxis]
            qx, qy, qz, qw = line[4:]
            # quat2mat accepts [w, x, y, z] quaternions
            Rsb = quat2mat([qw, qx, qy, qz])
            gsb = np.concatenate([Rsb, Tsb], axis=1)
            traj.append((ts, gsb))

        trajs.append(traj)


    # make sure they have same amount of entries
    assert len(trajs[0]) == len(trajs[1])

    # get rid of the zeros at the beginning
    size = len(trajs[0])
    for offset in range(size):
        # the 0-th field of each line is timestamp, 0 is invalid
        if trajs[0][offset][0] > 0 and trajs[1][offset][0] > 0:
            break

    trajs = [trajs[0][offset:], trajs[1][offset:]]
    size = len(trajs[0])

    pos = []
    for traj in trajs:
        pos.append(np.array([tup[1][:3, 3] for tup in traj]).transpose()) # 3xN

    # the transformation is from model (first argument) to data (second argument)
    # here it's from camera 1 to camera 0
    R01, T01, trans_error = align(np.matrix(pos[1]), np.matrix(pos[0]))
    R01, T01 = np.array(R01), np.array(T01)
    g01 = np.concatenate([R01, T01], axis=1)

    # fusion
    fused_traj = []
    for g0, g1 in zip(trajs[0], trajs[1]):
        t0, g0 = g0
        t1, g1 = g1
        R1_in_0 = g01[:3, :3].dot(g1[:3, :3])
        T1_in_0 = g01[:3, :3].dot(g1[:3, 3]) + g01[:3, 3]

        q0 = mat2quat(g0[:3, :3])
        q1_in_0 = mat2quat(R1_in_0)
        q = 0.5 * (q0 + q1_in_0)

        T = 0.5 * (g0[:3, 3] + T1_in_0)
        fused_traj.append([t0, T[0], T[1], T[2], q[3], q[0], q[1], q[2]])

    np.savetxt(
            os.path.join(args.working_dir, 'tumvi_{}_fused'.format(args.seq)),
            fused_traj,
            fmt='%f %f %f %f %f %f %f %f')




