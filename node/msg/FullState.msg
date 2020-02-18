std_msgs/Header            header

# These appear in the estimator state
geometry_msgs/Transform    gsb
geometry_msgs/Vector3      Vsb
geometry_msgs/Transform    gbc
geometry_msgs/Vector3      bg
geometry_msgs/Vector3      ba
geometry_msgs/Quaternion   qg
float64                    td
float64[9]                 Cg
float64[9]                 Ca
float64[9]                 camera_intrinsics
float64[]                  covariance


# Extra data, for convenience
uint32                     MotionStateSize
uint32                     group
geometry_msgs/Transform    gsc
bool                       MeasurementUpdateInitialized
geometry_msgs/Vector3      inn_Tsb
geometry_msgs/Vector3      inn_Wsb
geometry_msgs/Vector3      inn_Vsb