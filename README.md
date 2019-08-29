# XIVO: X Inertial-aided Visual Odometry


<!-- ![Demo](misc/demo.gif)

*Demo on TUM-VI room4 sequence. (downsampled video, not actual framerate)* -->

## Overview

XIVO is an open-source repository for visual-inertial odometry/mapping. It is a simplified version of Corvis, designed for pedagogical purposes, and incorporates odometry (relative motion of the sensor platform), local mapping (pose relative to a reference frame of the oldest visible features), and global mapping (pose relative to a global frame, including loop-closure and global re-localization — this feature not yet incorporated).

Corvis is optimized for speed, running at 200FPS on a commodity laptop computer, whereas XIVO runs at 100FPS. XIVO incorporates most of the core features of Corvis, including having 3D structure in the state, serving as short-term memory, performing auto-calibration (pose of the camera relative to the IMU, and time-stamp shift). It requires the camera to have calibrated intrinsics, which can be accomplished using any open-source package such as OpenCV prior to using Corvis. Corvis does require time-stamps, which can be done through the ROS drivers. Please refer to the ROS message interfaces ([imu][imu_msg],[image][image_msg]) for details on how to format the data for real-time use.

[imu_msg]: https://docs.ros.org/api/sensor_msgs/html/msg/Imu.html
[image_msg]: https://docs.ros.org/api/sensor_msgs/html/msg/Image.html

We provide several recorded sequences, and the ability to run XIVO off-line in batch mode, for comparison with other methods. Note that some of these methods operate in a non-causal fashion, by performing batch optimization relative to keyframe, or in a sliding window mode, introducing latency. XIVO is causal, and processes only the last image frame received.

Corvis has been developed since 2007, with contributors including Eagle Jones \[[ijrr11][jones_ijrr11]\], Konstantine Tsotsos \[[icra15][tsotsos_icra15]\], and Xiaohan Fei \[[cvpr17][dong_cvpr17],[eccv18][fei_eccv18],[icra19][fei_icra19]\]. If you use this code, or the datasets provided, please acknowledge it by citing \[[Fei *et al.*][fei_icra19]\].

While the ‘map’ produced by SLAM is only functional to localization, consisting of a sparse set of attributed point features, with the attributes sufficient for detection in an image, XIVO has been used as a building block for semantic mapping \[[Dong *et al.*][dong_cvpr17],[Fei *et al.*][fei_eccv18]\], where the scene is populated by objects, bounded by dense surfaces. Research code for semantic mapping using XIVO can be found [here][visma_repo].

[jones_ijrr11]: http://vision.ucla.edu/papers/jonesS10IJRR.pdf
[tsotsos_icra15]: http://vision.ucla.edu/papers/tsotsosCS15.pdf
[dong_cvpr17]: http://openaccess.thecvf.com/content_cvpr_2017/papers/Dong_Visual-Inertial-Semantic_Scene_Representation_CVPR_2017_paper.pdf
[fei_eccv18]: http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaohan_Fei_Visual-Inertial_Object_Detection_ECCV_2018_paper.pdf
[fei_icra19]: https://arxiv.org/abs/1807.11130v3
[visma_repo]: https://github.com/feixh/VISMA-tracker


## Background

The first public demonstration of real-time visual odometry on commercial off-the-shelf hardware was given by Hailin\[[cvpr00][hailin_cvpr00]\] at CVPR 2000. Its use for visual augmentation (augmented reality) was demonstrated at ICCV 2001\[[Favaro *et al.*][favaro_iccv01]\], and ECCV 2002. While visual odometry and SLAM are sometimes considered different, they are equivalent if structure is represented in the state and stored for later re-localization. This feature has been present in the work above since 2004 using feature group parametrization \[[Favaro *et al.*][favaro_iccv01]\]. Later public demonstrations of real-time visual odometry/SLAM include Andrew Davison \[[iccv03][davison_iccv03]\] and David Nister \[[cvpr04][nister_cvpr04]\].

Corvis is based on the analysis of [Jones-Vedaldi-Soatto][jonesVS_07] and was first demonstrated in 2008. The journal version of the paper describing the system was submitted in 2009 \[[ijrr][jones_ijrr11]\]. It differed from contemporaneous approaches using the MS-EKF in that it incorporated structure in the state of the filter, serving both as a reference for scale - not present in the MS-EKF - as well as a memory that enables mapping and re-localization, making it a hybrid odometry/mapping solution. One can of course also include in the model out-of-state feature constraints, in the manner introduced in the Essential Filter \[[Soatto 1996][soatto_eccv94]\](is this 94 or 96?), or the MS-EKF.

XIVO builds on Corvis, has features in the state and can incorporate out-of-state constraints and loop-closure, represents features in co-visibile groups, as in [Favaro][favaro_iccv01], and includes auto-calibration as in [Jones-Vedaldi-Soatto][jonesVS_07]. XIVO was also part of the first ever visual-inertial-semantic mapping in the literature [Dong-Fei][dong_cvpr17].

[soatto_eccv94]: https://link.springer.com/chapter/10.1007/BFb0028335
[hailin_cvpr00]: http://vision.ucla.edu/papers/jinFS00.pdf
[favaro_iccv01]: http://vision.ucla.edu/papers/favaroJS01DEMO.pdf
[davison_iccv03]: https://www.robots.ox.ac.uk/ActiveVision/Publications/davison_iccv2003/davison_iccv2003.pdf
[nister_cvpr04]: https://ieeexplore.ieee.org/abstract/document/1315094
[jonesVS_07]: http://vision.ucla.edu/papers/jonesVS07.pdf

## Requirements

This software is built and tested on Ubuntu 16.04 and 18.04 with compiler g++ 7.4.0.

## Dependencies

- [OpenCV][opencv]: Feature detection and tracking.
- [Eigen][eigen]: Linear algebra.
- [Pangolin][pangolin]: Lightweight visualization.
- [abseil-cpp][absl]: General utilities.
- [glog][glog]: Logging.
- [gflags][gflags]: Command-line options.
- [jsoncpp][jsoncpp]: Configuration.
- [pybind11][pybind11]: Python binding.
- (optional) [googletest][gtest]: Unit tests.
- (optional) [g2o][g2o]: To use pose graph optimization.
- (optional) [ROS][ros]: To use in live mode with ROS.

Dependencies are included in the `thirdparty` directory.

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

## Build

To build, in Ubuntu 18.04, execute the `build.sh` script in the root directory of the project. 

For detailed usage of the software, see the [wiki](https://github.com/feixh/xivo/wiki/HowTo).


## License and Disclaimer

This software is property of the UC Regents, and is provided free of charge for research purposes only. It comes with no warranties, expressed or implied, according to the terms linked [here](https://github.com/feixh/xivo/blob/master/LICENSE). For commercial use, please contact [UCLA TDG](https://tdg.ucla.edu).

## Acknowledgment

If you make use of any part of this code or the datasets provided, please acknowledge this repository by citing the follows:

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
