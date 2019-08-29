# Data Format

Each dataset contains two folders: One for the image sequence (named as `cam0`) and the other for the inertial measurements (named as `imu0`).

The `cam0` folder contains a `data` subfolder and a `data.csv` text file. The `data` subfolder contains a sequence of images named after the time instant (in nanoseconds as a 19-digit integer) when they are captured, and the text file contains two comma-separated columns of which the first column is the time-stamp in nanoseconds as a 19-digit integer, and the second column is the filename of the corresponding image captured at that time instant.


The `imu0` folder contains a single `data.csv` text file which contains 7 comma-separated columns with each row of the text file recording a time-stamped inertial measurement: The first column denotes the time-stamp when the inertial measurement was taken (in nanoseconds as a 19-digit integer), the second to the forth column represent the angular velocity (in rad/sec) measured by the IMU, and the last three columns record the linear acceleration (in m/sec<sup>2</sup>) measured by the IMU.