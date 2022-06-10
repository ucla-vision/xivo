import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import rosbag
from sensor_msgs.msg import Image, Imu, CameraInfo

parser = argparse.ArgumentParser("Plots allan variance vs. time interval length from a rosbag.")
parser.add_argument("--bag", type=str, help="rosbag containing IMU data")
parser.add_argument("--topic", type=str, default="/imu0", help="IMU data topic. Must have type sensor_msgs/imu")


def compute_variance_for_interval_length(interval_length_tsteps, signal):
  K = int(len(signal) / interval_length_tsteps)

  diffs = []

  start_idx = 0
  end_idx = interval_length_tsteps

  last_avg = 0
  for k in range(K):
    #curr_avg = np.mean(signal[start_idx:end_idx])
    curr_avg = (signal[end_idx] - signal[start_idx]) / interval_length_tsteps
    if k > 0:
      diffs.append((curr_avg - last_avg)**2)
    
    last_avg = curr_avg
    start_idx = end_idx
    end_idx = start_idx + interval_length_tsteps

  diffs = np.array(diffs)
  outval = 0.5*np.sum(diffs)
  return outval


def compute_avg_dt(time_axis):
  all_dt = np.zeros(len(time_axis)-1)
  for i in range(1, len(time_axis)):
    all_dt[i-1] = time_axis[i] - time_axis[i-1]
  return np.mean(all_dt)


def allan_vals(signal, interval_lengths_tsteps):
  allan_var = np.zeros(len(interval_lengths_tsteps))
  for i,interval_length in enumerate(interval_lengths_tsteps):
    allan_var[i] = compute_variance_for_interval_length(interval_length, signal)
  return allan_var    


def allan_xyz(signal_xyz, interval_lengths_secs, signal_type):
  dt = compute_avg_dt(time_axis)
  interval_lengths_tsteps = [ int(I / dt) for I in interval_lengths_secs ]

  allan_x = allan_vals(signal_xyz[:,0], interval_lengths_tsteps)
  allan_y = allan_vals(signal_xyz[:,1], interval_lengths_tsteps)
  allan_z = allan_vals(signal_xyz[:,2], interval_lengths_tsteps)
  plt.figure()
  plt.semilogy(interval_lengths_secs, allan_x)
  plt.semilogy(interval_lengths_secs, allan_y)
  plt.semilogy(interval_lengths_secs, allan_z)
  plt.legend(('x', 'y', 'z'))
  plt.ylabel("Allan Variance")
  plt.xlabel("Total  (s)")
  plt.title("{} Allan Variance".format(signal_type))


args = parser.parse_args()

bagfilename = args.bag
bagfile = rosbag.Bag(bagfilename, mode='r')

time_axis = []
gyro = []
accel = []


# Load real IMU messages into a numpy array
for _, msg, t in bagfile.read_messages(topics=args.topic):
  time_axis.append(t.to_sec())
  gyro.append([ msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z ])
  accel.append([ msg.linear_acceleration.x,
                 msg.linear_acceleration.y,
                 msg.linear_acceleration.z ])

time_axis = np.array(time_axis)
time_axis = time_axis - time_axis[0]

gyro = np.array(gyro)
accel = np.array(accel)

plt.figure()
plt.plot(time_axis, gyro[:,0])
plt.plot(time_axis, gyro[:,1])
plt.plot(time_axis, gyro[:,2])
plt.legend(("x", "y", "z"))
plt.title("gyro signals")

plt.figure()
plt.plot(time_axis, accel[:,0])
plt.plot(time_axis, accel[:,1])
plt.plot(time_axis, accel[:,2])
plt.legend(("x", "y", "z"))
plt.title("accel signals")

interval_lengths_secs = np.arange(1.0, 225.0, 1.0)

allan_xyz(gyro, interval_lengths_secs, "Gyroscope")
allan_xyz(accel, interval_lengths_secs, "Accelerometer")

plt.show()