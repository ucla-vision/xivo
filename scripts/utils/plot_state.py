import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    state = np.loadtxt('state.txt')
    estimate = np.loadtxt('estimate.txt')
    meas = np.loadtxt('meas.txt')
    plt.figure()
    tags = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']
    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            if idx < state.shape[1] and idx < estimate.shape[1]:
                plt.subplot(231+idx)
                plt.plot(state[:, idx], 'b*')
                plt.hold(True)
                plt.plot(estimate[:, idx], 'r.')
                plt.title(tags[idx])

    plt.figure()
    for i in range(meas.shape[1]):
        plt.subplot(101+meas.shape[1]*10+i)
        plt.hold(True)
        plt.plot(meas[:, i])
    plt.show()
