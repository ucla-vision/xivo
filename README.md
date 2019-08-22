# XIVO: X Inertial-aided Visual Odometry


![Demo](misc/demo.gif)

*Demo on TUM-VI room4 sequence. (downsampled video, not actual framerate)*

## Overview

XIVO is an open-source repository for visual-inertial odometry/mapping. It is a simplified version of Corvis, designed for pedagogical purposes, and incorporates odometry (relative motion of the sensor platform), local mapping (pose relative to a reference frame of the oldest visible features), and global mapping (pose relative to a global frame, including loop-closure and global re-localization — this feature not yet incorporated).

Corvis is optimized for speed, running at 200FPS on a commodity laptop computer, whereas XIVO runs at 100FPS. XIVO incorporates most of the core features of Corvis, including having 3D structure in the state, serving as short-term memory, performing auto-calibration (pose of the camera relative to the IMU, and time-stamp shift). It requires the camera to have calibrated intrinsics, which can be accomplished using any open-source package such as OpenCV prior to using Corvis. Corvis does require time-stamps, which can be done through the ROS drivers. Please refer to [link] for detailes on how to format the data for real-time use.

We provide several recorded sequences, and the ability to run XIVO off-line in batch mode, for comparison with other methods. Note that some of these methods operate in a non-causal fashion, by performing batch optimization relative to keyframe, or in a sliding window mode, introducing latency. XIVO is causal, and processes only the last image frame received.

Corvis has been developed since 2007, with contributors including Eagle Jones [cite IJRR paper], Konstantine Tsotsos [cite ICRA paper], and Xiaohan Fei [cite multiple papers]. If you use this code, or the datasets provided, please acknowledge it by citing [Fei].

While the ‘map’ produced by SLAM is only functional to localization, consisting of a sparse set of attributed point features, with the attributes sufficient for detection in an image, XIVO has been used as a building block for semantic mapping [cite], where the scene is populated by objects, bounded by dense surfaces. Research code for semantic mapping using XIVO can be found [here].


## Requirements

This software is built and tested on Ubuntu 16.04 and 18.04 with compiler g++ 7.4.0.

## Dependencies

- Pangolin:
- Eigen:
- g2o:
- abseil-cpp:
- googletest:
- glog:
- gflags:
- jsoncpp:
- pybind11:

Dependencies are included in the `thirdparty` directory.

## Build

To build, in Ubuntu 18.04, execute the `build.sh` script in the root directory of the project. 

For detailed usage of the software, see the [wiki](https://github.com/feixh/xivo/wiki/HowTo).


## License and Disclaimer

This software is property of the UC Regents, and is provided free of charge for research purposes only. It comes with no warranties, expressed or implied, according to the terms linked [here](https://github.com/feixh/xivo/edit/master/LICENSE). For commercial use, please contact [UCLA TDG](tdg.ucla.edu).

## Acknowledgment

If you make use of any part of this code or the datasets provided, please acknoledge this repository by citing the follows:

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


