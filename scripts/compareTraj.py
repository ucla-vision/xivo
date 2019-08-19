"""
Head-to-head comparison of MoCap ground truth and estimated state.
Usage: python compareTraj.py SEQUENCE_NAME YOUR_OUTPUT_FILE
If not working, change the path template
Author: Xiaohan Fei
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

# path_template = '/home/feixh/Data/tumvi/exported/euroc/512_16/dataset-{}_512_16/mav0/mocap0/data.csv'
path_template = '/local2/Data/tumvi/exported/euroc/512_16/dataset-{}_512_16/mav0/mocap0/data.csv'
# path_template = '/local/Data/tumvi/exported/euroc/512_16/dataset-{}_512_16/mav0/mocap0/data.csv'

def loadMoCapData(seq):
    if seq == '-':
        return None, None
    data = []
    with open(path_template.format(seq), 'r') as fid:
        for l in fid.readlines()[1:]:
            data.append([float(x) for x in l.strip().split(',')])
    data = np.asarray(data)

    ts = data[:, 0]
    T = data[:, 1:4]
    return ts, T

def loadResults(path):
    if path == '-':
        return None, None
    data = np.loadtxt(path)
    ts = data[:, 0]
    T = data[:, 1:4]
    return ts, T

if __name__ == '__main__':
    mocap_ts, mocap_T = loadMoCapData(sys.argv[1])
    est_ts, est_T = loadResults(sys.argv[2])

    valid = est_ts > 0
    est_ts = est_ts[valid]
    est_T = est_T[valid, :]

    tag='xyz'
    t0 = min(mocap_ts[0], est_ts[0])
    for i in range(3):
        plt.subplot(311+i)
        if mocap_ts is not None:
            plt.plot((mocap_ts - t0) * 1e-9, mocap_T[:, i], 'b', label='MoCap')
        if est_ts is not None:
            plt.plot((est_ts - t0) * 1e-9, est_T[:, i], 'r', label='MSCKF')
        plt.title('translation-' + tag[i])
        plt.legend()
    plt.show()
