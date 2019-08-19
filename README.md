# X Inertial-aided Visual Odometry

I started writing this software as a hobby project in winter 2018. The motivation behind this is in part that I want to have a (hopefully) minimal implementation of EKF-based Visual-Inertial Odometry and to play with some new C++ language features (C++ 14/17) and coding techniques (templates, design patterns, etc.). The project was paused for months due to job hunting, paper submission, defense and vacation, etc. Fortunately, I was able to spend more time on it before I leave UCLA.

This is still work in progress, and under no circumstances, it can be considered as done. However, with readability and maintainability in mind when writing this piece of software, I believe it still has some value for people who want to dive into the world of VIO. Besides, the performance of the system in terms of ATE (Absolute Trajectory Error) and RPE (Relative Pose Error) is comparable to other open-source VIO systems (see the performance section). 


A detailed technical report of the implementation will follow. Stay tuned.

Have fun!

## Install

To install the state estiamtor and dependencies, execute the `build.sh` script in the root directory of the project.

## Usage

The current implementation supports two executation modes: Either run on a folder of image sequences and a text file of inertial measurements (like what provided by the EuRoC and TUM-VI datasets), or a rosbag (TUM-VI also provides rosbags).

### Dataset

Assume the environment variable `$TUMVIROOT` has been set to the root directory of your [TUMVI](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) dataset. For example,

```
export TUMVIROOT=/home/Data/tumvi/exported/euroc/512_16
```

where on my machine `/home/Data/tumvi/exported/euroc/512_16` hosts folders of data, such as `dataset-room1_512_16`, `dataset-corridor1_512_16`, etc.

### Run

In the project root directory, you can run the estimator with configuration specified by the `-cfg cfg/vio.json` option on images captured by camera 0 (`-cam_id 0`) of 6-th room sequence (`-seq room6`) of the TUMVI dataset (`-dataset tumvi`) which resides in `$TUMVIROOT` (option `-root $TUMVIROOT`) as follows:

```
bin/st_vio -cfg cfg/vio.json -root $TUMVIROOT -dataset tumvi -seq room6 -cam_id 0 -out out_state
```

where estimated states are saved to the output file `out_state`.

For detailed usage of the application, see the flags defined at the begining of the application source file `src/app/singlethread_vio.cpp`. 


### Evaluation

<!--
**DEPRECATED: USE THE PYTHON EVALUATION SCRIPT FROM TUM RGBD BENCHMARK INSTEAD**


A C++ implementation of RPE (Relative Pose Error) and ATE (Absolute Trajectory Error) is provided. Once you have run the estimator on one of the TUM-VI sequence and have the estimated states saved in a file, say, `out_state`, evaluation can be done as follows:

```
bin/eval -root $TUMVIROOT -seq room6 -result out_state -resolution 0.001
```

where `resolution` specifies the temporal interval over which RPE is measured.
-->

We provide a python script `scripts/run_and_eval_pyxivo.py` to run the estimator on a specified TUM-VI sequence and benchmark the performance in terms ATE (Absolute Trajectory Error) and RPE (Relative Pose Error). To use it, execute the following in the project root directory:

```
python scripts/run_and_eval_pyxivo.py -root $TUMVIROOT -seq room6 -stdout -out_dir tmp -use_viewer
```
the `-seq` and `-root` options are the same as explained above. If the `-stdout` option is on, the script will print out the benchmarked performance to the terminal; `-out_dir` specifies the directory to save state estimates; `-use_viewer` option will turn on a 3D visualization. For detailed usage about the script, see the options defined at the begining of the script.

### Log to file for debugging

We use [glog](https://github.com/google/glog) for system logging. If you want to log debug information to a file for inspection later,

```
GLOG_log_dir=log GLOG_v=0 bin/st_vio -cfg cfg/vio.json -root $TUMVIROOT
```

Note, the directory `log`, which is where the logs are kept, should be created first (simply `mkdir log`). 

By default, log is suppressed by setting `add_definitions(-DGOOGLE_STRIP_LOG=1)` in `src/CMakeLists.txt`. To enable log, simply comment out that line and re-build.

For more details on how to use glog, see [the tutorial here](http://rpg.ifi.uzh.ch/docs/glog.html).

<!--
### Python scripts for diagnosis

We provide some python scripts in `scripts` folder to diagnose the behavior of the system. By default, the vio application will dump state to a text file named `out_state_aligned`. To compare the state estimates and the ground-truth states, you can do the following:

```
python scripts/compareTraj.py room6 out_state_aligned
```

Note, you can change `out_state_aligned` to another name as long as it is consistent with the filename you provide to the `-out` option when your run the vio application. Also `room6` is the sequence name to perform the comparison, and it should be the same as the one used when you run your application.

If the script works fine, you will be able to see a figure of three panels with each of which showing the x, y and z components of the estimated translation state. The blue trajectory is the ground truth, and the red one is your estimation.
-->

## Rosbag mode

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

An example of using the python binding is available in `scripts/pyxivo.py`, which demonstrates estimator creation, data loading, and visualization in python.

To run the demo, simply execute:

```
python scripts/pyxivo.py -root $TUMVIROOT -seq room6 -cam_id 0
```

in the project root directory. The command line options are more or less same as the C++ executable. For detailed usage, you can look at the options defined at the begining of the script `scripts/pyxivo.py`. Note you might need to install some python dependencies:

```
pip install -r requirements.txt
```

executed in the project root directory.

## Performance

The benchmark performance of this software on TUM-VI dataset is comparable to other open-source VIO systems (slightly worse than optimization-based OKVIS and VINS-Mono and on par with EKF-based ROVIO). Also, our system runs at more than 100 Hz on a desktop PC with a Core i7 6th gen CPU at very low CPU consumption rate. The runtime can be further improved by utilizing CPU cache and memory better. The following table shows the performance on 6 indoor sequences where ground-truth poses are available. The numbers for OKVIS, VINS-Mono and ROVIO are taken from the TUM-VI benchmark paper. Ours XIVO is obtained by using the aforementioned evaluation script.


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


---

## Reference

If you find this software useful and use it in your work, please cite:

```
@misc{feiS19xivo,
  author = {Xiaohan Fei, and Stefano Soatto},
  title = {XIVO: X Inertial-aided Visual Odometry},
  howpublished = "\url{https://github.com/feixh/xivo}",
}
```

<!--
and related papers which this implementation is based on:

```
@article{jones2011visual,
  title={Visual-inertial navigation, mapping and localization: A scalable real-time causal approach},
  author={Jones, Eagle S and Soatto, Stefano},
  journal={The International Journal of Robotics Research},
  volume={30},
  number={4},
  pages={407--430},
  year={2011},
  publisher={SAGE Publications Sage UK: London, England}
}
```
or 
```
@inproceedings{tsotsos2015robust,
  title={Robust inference for visual-inertial sensor fusion},
  author={Tsotsos, Konstantine and Chiuso, Alessandro and Soatto, Stefano},
  booktitle={2015 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={5203--5210},
  year={2015},
  organization={IEEE}
}
```

-->
