import numpy as np
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import sys
sys.path.append("scripts")
from imu_sim import Havertrig1d, QuaternionSlew, qdot
from pltutils import time_n_plots


def havertrig_deriv(t: float, state: np.ndarray, accel_func):
    x = state[0]
    v = state[1]
    deriv = np.zeros(2)
    deriv[0] = v
    deriv[1] = accel_func(t)
    return deriv


def test_havertrig(x0: float, xf: float, T: float):
    all_t = np.arange(0.0, T, 0.01)
    slewer = Havertrig1d(x0, xf, T)
    deriv_func = lambda t,x: havertrig_deriv(t, x, slewer.accel)
    output = solve_ivp(deriv_func, [0.0, T], [x0, 0.0], t_eval=all_t)
    t_int = output.t
    position_vals = output.y[0,:]

    plt.figure()
    plt.plot(t_int, position_vals)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (s)")
    plt.title("{:.1f} -> {:.1f} in {:.1f} sec".format(x0, xf, T))


def quat_deriv(t: float, q: np.ndarray, omega_func):
    omega = omega_func(t)
    q_deriv = qdot(q, omega)
    return q_deriv


def test_qslew(q0: np.ndarray, q1: np.ndarray, T: float):
    all_t = np.arange(0.0, T, 0.001)
    slewer = QuaternionSlew(q0, q1, T)

    deriv_func = lambda t,q: quat_deriv(t, q, slewer.omega)
    output = solve_ivp(deriv_func, [0.0, T], q0, t_eval=all_t)
    t_int = output.t
    q_vals = output.y

    suptitle_str = "Slew: [{:.2f}, {:.2f}, {:.2f}, {:.2f}] ".format(
        q0[0], q0[1], q0[2], q0[3]
    ) + "-> [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
        q1[0], q1[1], q1[2], q1[3]
    )
    time_n_plots(t_int, q_vals, suptitle_str, xlabel="Time (s)",
                 ylabels=[ "x", "y", "z", "w"])

    """
    omega_vals = np.zeros((3, len(t_int)))
    for i,t in enumerate(t_int):
        omega_vals[:,i] = omega_func(t)
    time_n_plots(t_int, omega_vals, "Angular velocity", xlabel="Time (s)",
                 ylabels=[ "x", "y", "z"])

    # Slerp results
    q_slerped = np.zeros((4, len(t_int)))
    for i,t in enumerate(t_int):
        q_slerped[:,i] = slewer.slerp(t)
    time_n_plots(t_int, q_slerped, "debug quaternions", xlabel="Time (s)",
                 ylabels=[ "x", "y", "z", "w"])

    # slerpdot output
    slerpdot_output = np.zeros((4, len(t_int)))
    for i,t in enumerate(t_int):
        slerpdot_output[:,i] = slewer.slerp_dot(t)
    time_n_plots(t_int, slerpdot_output, "debug slerp dot", xlabel="Time (s)",
                 ylabels=["x", "y", "z", "w"])

    # integrate the area under the curve of qdot
    slerpdot_integrated = np.trapz(slerpdot_output, t_int)
    print("Integrated qdot: {}".format(slerpdot_integrated))
    print("begin + diff: {}".format(q0  + slerpdot_integrated))
    print("end: {}".format(q1))
    """



if __name__ == "__main__":
    test_havertrig(10.0, 20.0, 5.0)
    test_havertrig(30.0, 5.0, 25.0)

    q0a = np.array([0, 0, 0, 1])
    q1a = Rotation.from_euler('XYZ', [np.pi/2, 0, 0]).as_quat()
    test_qslew(q0a, q1a, 10.0)

    q0b = np.array([0.2, -0.3, 0.4, 0.6])
    q0b = q0b / np.linalg.norm(q0b)
    q1b = np.array([1.0, 2.0, 3.0, 4.0])
    q1b = q1b / np.linalg.norm(q1b)
    test_qslew(q0b, q1b, 15.0)

    plt.show()