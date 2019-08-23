# XIVO: X Inertial-aided Visual Odometry


![Demo](misc/demo.gif)

*Demo on TUM-VI room4 sequence. (downsampled video, not actual framerate)*

## Overview

XIVO is an open-source repository for visual-inertial odometry/mapping. It is a simplified version of Corvis, designed for pedagogical purposes, and incorporates odometry (relative motion of the sensor platform), local mapping (pose relative to a reference frame of the oldest visible features), and global mapping (pose relative to a global frame, including loop-closure and global re-localization — this feature not yet incorporated).

Corvis is optimized for speed, running at 200FPS on a commodity laptop computer, whereas XIVO runs at 100FPS. XIVO incorporates most of the core features of Corvis, including having 3D structure in the state, serving as short-term memory, performing auto-calibration (pose of the camera relative to the IMU, and time-stamp shift). It requires the camera to have calibrated intrinsics, which can be accomplished using any open-source package such as OpenCV prior to using Corvis. Corvis does require time-stamps, which can be done through the ROS drivers. Please refer to [link] for detailes on how to format the data for real-time use.

We provide several recorded sequences, and the ability to run XIVO off-line in batch mode, for comparison with other methods. Note that some of these methods operate in a non-causal fashion, by performing batch optimization relative to keyframe, or in a sliding window mode, introducing latency. XIVO is causal, and processes only the last image frame received.

Corvis has been developed since 2007, with contributors including Eagle Jones [cite IJRR paper], Konstantine Tsotsos [cite ICRA paper], and Xiaohan Fei [cite multiple papers]. If you use this code, or the datasets provided, please acknowledge it by citing [Fei].

While the ‘map’ produced by SLAM is only functional to localization, consisting of a sparse set of attributed point features, with the attributes sufficient for detection in an image, XIVO has been used as a building block for semantic mapping [cite], where the scene is populated by objects, bounded by dense surfaces. Research code for semantic mapping using XIVO can be found [here].

## Background

The first public demonstration of real-time visual odometry on commercial off-the-shelf hardware was given by [Hailin] at CVPR 2000. It use for visual augmentation (augmented reality) was demonstrated at ICCV 2001[Favaro], and ECCV 2002. While visual odometry and SLAM are sometimes considered different, they are equivalent if structure is represented in the state and stored for later re-localization. This feature has been present in the work above since 2004 using feature group parametrization [Favaro]. Later public demonstrations of real-time visual odometry/SLAM include [Andrew Davison] and [David Nister, 2005].

Corvis is based on the analysis of [Jones-Vedaldi-Soatto] and was first demonstrated in 2008. The journal version of the paper describing the system was submitted in 2009 [IJRR]. It differed from contemporaneous approaches using the MS-EKF in that it incorporated structure in the state of the filter, serving both as a reference for scale - not present in the MS-EKF - as well as a memory that enables mapping and re-localization, making it a hybrid odometry/mapping solution. One can of course also include in the model out-of-state feature constraints, in the manner introduced in the Essential Filter [Soatto 1996], or the MS-EKF.

XIVO builds on Corvis, has features in the state and can incorporate out-of-state constraints and loop-closure, represents features in co-visibile groups, as in [Favaro], and includes auto-calibration as in [Jones-Vedaldi]. XIVO was also part of the first ever visual-inertial-semantic mapping in the literature [Jingming-Xiaon].


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

<!--

## Usage

The current implementation supports two execution modes: Batch, requiring a folder of image sequences and a text file of time-stamped inertial measurements, or online using a ROS bag. The format for the image sequences and text files are described [here](placeholder).

<!--
Either run on a folder of image sequences and a text file of inertial measurements (like what provided by the EuRoC and TUM-VI datasets), or a rosbag (TUM-VI also provides rosbags).
-->

### Datasets

- XIVO 1:
- XIVO 2:
...
- TUM-VI:

<!--
Assume the environment variable `$TUMVIROOT` has been set to the root directory of your [TUMVI](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) dataset. For example,

```
export TUMVIROOT=/home/Data/tumvi/exported/euroc/512_16
```

where on my machine `/home/Data/tumvi/exported/euroc/512_16` hosts folders of data, such as `dataset-room1_512_16`, `dataset-corridor1_512_16`, etc.
-->

### Run

From the project root directory, run the estimator with configuration specified by the `-cfg cfg/vio.json` option on images captured by camera 0 (`-cam_id 0`) of the 6-th room sequence (`-seq room6`) of the TUMVI dataset (`-dataset tumvi`) which resides in `$TUMVIROOT` (option `-root $TUMVIROOT`) as follows:

```
bin/st_vio -cfg cfg/vio.json -root $TUMVIROOT -dataset tumvi -seq room6 -cam_id 0 -out out_state
```

where estimated states are saved to the output file `out_state`.

For detailed usage of the application, see the flags defined at the beginning of the application source file `src/app/singlethread_vio.cpp`. 


### Evaluation

We provide a python script `scripts/run_and_eval_pyxivo.py` to run the estimator on a specified TUM-VI sequence and benchmark the performance in terms ATE (Absolute Trajectory Error) and RPE (Relative Pose Error). To use it, execute the following in the project root directory:

```
python scripts/run_and_eval_pyxivo.py -root $TUMVIROOT -seq room6 -stdout -out_dir tmp -use_viewer
```
The `-seq` and `-root` options are the same as explained above. If the `-stdout` option is on, the script will print out the benchmarked performance to the terminal; `-out_dir` specifies the directory to save state estimates; `-use_viewer` option will turn on a 3D visualization. For detailed usage about the script, see the options defined at the beginning of the script.

### Log to file for debugging

We use [glog](https://github.com/google/glog) for system logging. If you want to log debug information to a file for inspection later,

```
GLOG_log_dir=log GLOG_v=0 bin/st_vio -cfg cfg/vio.json -root $TUMVIROOT
```

Note, the directory `log`, which is where the logs are kept, should be created first (simply `mkdir log`). 

By default, log is suppressed by setting `add_definitions(-DGOOGLE_STRIP_LOG=1)` in `src/CMakeLists.txt`. To enable log, simply comment out that line and re-build.

For more details on how to use glog, see [the tutorial here](http://rpg.ifi.uzh.ch/docs/glog.html).

## ROS bag mode

### Build

By default the ROS wrapper of the system will not be built, to build the ROS support, you need to turn on the `BUILD_ROSNODE` option when generating the makefile: In `build` directory of the project root, do the following:

```
cmake .. -DBUILD_ROSNODE=ON
```

followed by `make` to build with ROS support.

### Run

1. In the project root directory, `source build/devel/setup.zsh`. If another shell, such as bash/sh, is used, please source the corresponding shell script (`setup.bash`/`setup.sh`) respectively.
2. `roslaunch node/launch/xivo.launch` to launch the ros node, which subscribes to `/imu0` and `/cam0/image_raw` topics.
3. In antoher terminal, playback rosbags from the TUM-VI dataset, e.g., `rosbag play PATH/TO/YOUR/ROSBAG` .

<!-- ## Profiling

If you want to build the project along with the gperftools provided in the thirdparty folder, make sure you have `autoconf` and `libtool` installed.
`
sudo apt-get install autoconf libtool
`
and
`
./build.sh
`

See [gperftools](https://gperftools.github.io/gperftools/cpuprofile.html) from Google. Or enable printing of the timing information gathered by the `timer_` object inside the estimator. -->


## Python binding

A simple python binding is provided by wrapping some public interfaces of the estimator via [`pybind11`](https://github.com/pybind/pybind11). Check out `pybind11/pyxivo.cpp` for the available interfaces in python. With pybind11, it is relatively easy if you want to expose more interfaces of the C++ implementation to python.

An example of using the Python binding is available in `scripts/pyxivo.py`, which demonstrates estimator creation, data loading, and visualization in python.

To run the demo, execute:

```
python scripts/pyxivo.py -root $TUMVIROOT -seq room6 -cam_id 0
```

in the project root directory. The command-line options are more or less the same as the C++ executable. For detailed usage, you can look at the options defined at the beginning of the script `scripts/pyxivo.py`. Note you might need to install some python dependencies by executing the following in the project root directory:

```
pip install -r requirements.txt
```

## Performance

The benchmark performance of this software on TUM-VI dataset is comparable to other open-source VIO systems. Also, our system runs at more than 100 Hz on a desktop PC with a Core i7 6th gen CPU at very low CPU consumption rate. The runtime can be further improved by utilizing CPU cache and memory better. The following table shows the performance on 6 indoor sequences where ground-truth poses are available. The numbers for OKVIS, VINS-Mono, and ROVIO are taken from the TUM-VI benchmark paper. Ours XIVO is obtained by using the aforementioned evaluation script.


| Sequence | length | OKVIS | VINS-Mono | ROVIO | Ours-XIVO |
|:---       | :---    | :---:   | :---:       | :---:   | :---:  |
|room1     | 156m   | **0.06m** | 0.07m | 0.16m | 0.22m |
|room2     | 142m   | 0.11m | **0.07m** | 0.33m | 0.08m |
|room3     | 135m   | **0.07m**  | 0.11m | 0.15m | 0.11m |
|room4     | 68m    | **0.03m** | 0.04m | 0.09m | 0.08m |
|room5     | 131m   | **0.07m** | 0.20m | 0.12m | 0.11m |
|room6     | 67m    | **0.04m** | 0.08m | 0.05m | 0.12m |

*Table 1. RMSE ATE* in meters. OKVIS and VINS-Mono are optimization-based, whereas ROVIO and Ours-XIVO are EKF-based.


| Sequence | OKVIS | VINS-Mono | ROVIO | Ours-XIVO |
|:---       | :---:   | :---:       | :---:   | :---:  |
|room1 | **0.013**m/**0.43**<sup>o</sup> | 0.015m/0.44<sup>o</sup> | 0.029m/0.53<sup>o</sup> | 0.031m/0.59<sup>o</sup> |
|room2 | **0.015**m/**0.62**<sup>o</sup> | 0.017m/0.63<sup>o</sup> | 0.030m/0.67<sup>o</sup> | 0.023m/0.75<sup>o</sup> |
|room3 | **0.012**m/0.64<sup>o</sup> | 0.023m/**0.63**<sup>o</sup> | 0.027m/0.66<sup>o</sup> | 0.027m/0.73<sup>o</sup> |
|room4 | **0.012**m/0.57<sup>o</sup> | 0.015m/**0.41**<sup>o</sup> | 0.022m/0.61<sup>o</sup> | 0.023m/0.62<sup>o</sup> |
|room5 | **0.012**m/**0.47**<sup>o</sup> | 0.026m/**0.47**<sup>o</sup> | 0.031m/0.60<sup>o</sup> | 0.023m/0.65<sup>o</sup> |
|room6| **0.012**m/0.49<sup>o</sup> | 0.014m/**0.44**<sup>o</sup> | 0.019m/0.50<sup>o</sup> | 0.031m/0.53<sup>o</sup> |

*Table 2. RMSE RPE* in meters (translation) and degrees (rotation).



-->
