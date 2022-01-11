# XIVO: X Inertial-aided Visual Odometry and Sparse Mapping


![Demo](misc/demo_ucla_e6.gif)

XIVO is an open-source repository for visual-inertial odometry/mapping. It is a simplified version of Corvis \[[Jones *et al.*][jones_ijrr11],[Tsotsos *et al.*][tsotsos_icra15]\], designed for pedagogical purposes, and incorporates odometry (relative motion of the sensor platform), local mapping (pose relative to a reference frame of the oldest visible features), and global mapping (pose relative to a global frame, including loop-closure and global re-localization — this feature, present in Corvis, is not yet incorporated in XIVO).

XIVO runs at 140FPS on stored data (here from a RealSense D435i sensor) or on live streams with latency of around 1-7ms, depending on the hardware. It takes as input video frames from a calibrated camera and inertial measurements from an IMU, and outputs a sparse point cloud with attribute features and 6 DOF pose of the camera. It performs auto-calibration of the relative pose between the camera and the IMU as well as the time-stamp alignment. More demos are available [here](demo.md), the aproach is described in this [paper][tsotsos_icra15]. XIVO does not perform post-mortem refinement (bundle adjustment, pose graph optimization), but that can be easily added as post-processing.


## Requirements

This software is primarily built and tested on Ubuntu 20.04 with compiler g++9. We may support other platforms as we feel like. A full list of supported platforms is listed in our [build page](https://github.com/ucla-vision/xivo/wiki/Build-Instructions).


## Dependencies

- [OpenCV][opencv]: Feature detection and tracking.
- [Eigen][eigen]: Linear algebra.
- [Pangolin][pangolin]: Lightweight visualization.
- [glog][glog]: Logging.
- [gflags][gflags]: Command-line options.
- [jsoncpp][jsoncpp]: Configuration.
- (optional) [googletest][gtest]: Unit tests.
- (optional) [g2o][g2o]: To use pose graph optimization.
- (optional) [ROS][ros]: To use in live mode with ROS.
- (optional) [pybind11][pybind11]: Python binding.
<!-- - [abseil-cpp][absl]: General utilities. -->

All dependencies, except for OpenCV, are included in the `thirdparty` directory.

[opencv]: https://opencv.org/
[eigen]: http://eigen.tuxfamily.org/index.php?title=Main_Page
[g2o]: https://github.com/RainerKuemmerle/g2o
[pangolin]: https://github.com/stevenlovegrove/Pangolin
[absl]: https://abseil.io/
[gtest]: https://github.com/google/googletest
[glog]: https://github.com/google/glog
[gflags]: https://github.com/gflags/gflags
[jsoncpp]: https://github.com/open-source-parsers/jsoncpp
[pybind11]: https://github.com/pybind/pybind11
[ros]: https://www.ros.org/


## Build and Usage

Please see our [wiki](https://github.com/ucla-vision/xivo/wiki) for usage instructions and more detailed information about the algorithm.


## License and Disclaimer

This software is property of the UC Regents, and is provided free of charge for research purposes only. It comes with no warranties, expressed or implied, according to these [terms and conditions](LICENSE). For commercial use, please contact [UCLA TDG](https://tdg.ucla.edu).

## <a name="ack-anchor"></a> Acknowledgment



If you make use of any part of this code or the datasets provided, please acknowledge this repository by citing the following:
```
@misc{fei2019xivo,
title={XIVO: An Open-Source Software for Visual-Inertial Odometry},
author={Fei, Xiaohan and Soatto, Stefano},
year={2019},
howpublished = "\url{https://github.com/ucla-vision/xivo}"
}
```
or

```
@article{fei2019geo,
  title={Geo-supervised visual depth prediction},
  author={Fei, Xiaohan and Wong, Alex and Soatto, Stefano},
  journal={IEEE Robotics and Automation Letters},
  volume={4},
  number={2},
  pages={1661--1668},
  year={2019},
  publisher={IEEE}
}
```


