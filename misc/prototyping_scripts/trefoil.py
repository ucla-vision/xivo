import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import sys
sys.path.append("scripts")
from pltutils import time_three_plots

t = np.linspace(0, 100.0, 1000)

x = (4 + np.cos(3*t)) * np.cos(2*t) - (4+1)*1
y = (4 + np.cos(3*t)) * np.sin(2*t)
z = np.sin(3*t)

xd = -2*np.sin(2*t)*(np.cos(3*t) + 4) - 3*np.cos(2*t) * np.sin(3*t)
yd = 2*np.cos(2*t)*(np.cos(3*t) + 4) - 3*np.sin(2*t) * np.sin(3*t)
zd = 3 * np.cos(3*t)

xdd = 12*np.sin(2*t)*np.sin(3*t) - 9*np.cos(2*t)*np.cos(3*t) - 4*np.cos(2*t)*(np.cos(3*t)+4)
ydd = -4*np.sin(2*t)*(np.cos(3*t)+4) - 12*np.cos(2*t)*np.sin(3*t) - 9*np.cos(3*t)*np.sin(2*t)
zdd = -9.0 * np.sin(3*t)


def deriv(t, x):
    vel = x[3:6]
    dxdt = np.zeros(6)
    dxdt[0:3] = vel
    dxdt[3] = 12*np.sin(2*t)*np.sin(3*t) - 9*np.cos(2*t)*np.cos(3*t) - 4*np.cos(2*t)*(np.cos(3*t)+4)
    dxdt[4] = -4*np.sin(2*t)*(np.cos(3*t)+4) - 12*np.cos(2*t)*np.sin(3*t) - 9*np.cos(3*t)*np.sin(2*t)
    dxdt[5] = -9.0 * np.sin(3*t)
    return dxdt
   

#def vel(t, x):
#    xd = -2*np.sin(2*t)*(np.cos(3*t) + 4) - 3*np.cos(2*t) * np.sin(3*t)
#    yd = 2*np.cos(2*t)*(np.cos(3*t) + 4) - 3*np.sin(2*t) * np.sin(3*t)
#    zd = 3 * np.cos(3*t)
#    return np.array([xd, yd, zd])
#
#ic = np.array([x[0], y[0], z[0]])
#output = solve_ivp(vel, [0, 100.0], ic, t_eval=t)

ic = np.array([x[0], y[0], z[0], xd[0], yd[0], zd[0]])
output = solve_ivp(deriv, [0, 100.0], ic, t_eval=t)
t_int = output.t
x0 = output.y[0,:]
y0 = output.y[1,:]
z0 = output.y[2,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

print("initial position: {} {} {}".format(x[0], y[0], z[0]))
print("initial velocity: {} {} {}".format(xd[0], yd[0], zd[0]))

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111, projection='3d')
#ax2.plot(x0, y0, z0)

#time_three_plots(t, np.vstack((x, y, z,)), "trefoil positions")
#time_three_plots(t, np.vstack((xd, yd, zd)), "trefoil velocities")
#time_three_plots(t, np.vstack((xdd, ydd, zdd)), "trefoil accelerations")

plt.show()