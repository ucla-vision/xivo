std_msgs/Header            header

# These appear in the estimator state (includes temporal and IMU calibration)
geometry_msgs/Transform    gsb
geometry_msgs/Vector3      Vsb
geometry_msgs/Transform    gbc
geometry_msgs/Vector3      bg
geometry_msgs/Vector3      ba
geometry_msgs/Quaternion   qg
float64                    td
float64[9]                 Cg
float64[9]                 Ca
float64[]                  covariance

# Online camera calibration
# camera parameters are always in the order
# fx, fy, cx, cy, [distortion params]
# For equi, [distortion params] = k0123
# For radtan, [distortion params] = p1, p2, k1, k2, k3
# if there are less than five distortion parameters, then the rest will be zero
string                     projection_model   # for now, always "pinhole"
string                     distortion_model   # "none", "radtan", "equi"
uint32                     num_camera_params  # min 4, max 9
float64[9]                 camera_intrinsics
float64[]                  camera_covariance

# Extra data, for convenience
uint32                     MotionStateSize
uint32                     group
geometry_msgs/Transform    gsc
bool                       MeasurementUpdateInitialized
geometry_msgs/Vector3      inn_Tsb
geometry_msgs/Vector3      inn_Wsb
geometry_msgs/Vector3      inn_Vsb