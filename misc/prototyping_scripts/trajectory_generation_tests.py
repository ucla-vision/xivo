import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import sys
sys.path.append("scripts")
from imu_sim import havertrig_accel_1d


def havertrig_deriv(t: float, state: np.ndarray, accel_func):
    x = state[0]
    v = state[1]
    deriv = np.zeros(2)
    deriv[0] = v
    deriv[1] = accel_func(t)
    return deriv


def test_havertrig(x0: float, xf: float, T: float):
    all_t = np.arange(0.0, T, 0.01)
    accel_func = lambda t: havertrig_accel_1d(t, x0, xf, T)
    deriv_func = lambda t,x: havertrig_deriv(t, x, accel_func)
    output = solve_ivp(deriv_func, [0.0, T], [x0, 0.0], t_eval=all_t)
    t_int = output.t
    position_vals = output.y[0,:]

    plt.figure()
    plt.plot(t_int, position_vals)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (s)")
    plt.title("{:.1f} -> {:.1f} in {:.1f} sec".format(x0, xf, T))



if __name__ == "__main__":
    test_havertrig(10.0, 20.0, 5.0)
    test_havertrig(30.0, 5.0, 25.0)

    plt.show()