# This contains the motion state for a 2d navigation problem. All coordinates
# are in the "gravity frame", the frame where gravity = [0, 0, -9.8], rather
# than the IMU frame. The gravity frame shares an origin with the IMU frame.
# In the gravity frame, the yaw angle is the heading. Heading is equal to 0
# at initialization.
#
# Because the gravity frame shares an origin with the spatial frame, coordinate
# transformations from the spatial frame are simple.


std_msgs/Header         header

geometry_msgs/Point     position
geometry_msgs/Vector3   velocity
float64[9]              position_covariance
float64[9]              velocity_covariance

# euler_angles = [roll, pitch, yaw]
geometry_msgs/Vector3   euler_angles