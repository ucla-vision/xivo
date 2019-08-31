## Usage

The current implementation supports two execution modes: Batch, requiring a folder of image sequences and a text file of time-stamped inertial measurements, or online using a ROS bag. The format for the image sequences and text files are described [here](dataformat.md). The format of the ROS messages is described here ([imu][imu_msg],[image][image_msg]).

[imu_msg]: https://docs.ros.org/api/sensor_msgs/html/msg/Imu.html
[image_msg]: https://docs.ros.org/api/sensor_msgs/html/msg/Image.html


### Datasets

We provide several sequences recorded by a [Tango](https://en.wikipedia.org/wiki/Tango_(platform)) platform. The images are recorded at 30 Hz in VGA (640x480) size, and the inertial measurements are recorded at 200 Hz. You can download the sequences [here][xivo_download] as compressed files (`.tar.gz`) or ROS bags (`.bag`). We will add more sequences to the dataset as well as metric way points for thorough evaluation of VIO systems.

[xivo_download]: https://www.dropbox.com/sh/0w5b7fglxf3li2l/AABAGYTU8QCq-vPuD-cqO4xta?dl=0

### Setup

Once you have downloaded the data as compressed `.tar.gz` files, uncompress them into the directory of your choice and set the environment variable `$DATAROOT` to that directory as the following:

```
export DATAROOT=/DIRECTORY/OF/YOUR/CHOICE
```

Note, the `$DATAROOT` directory should contain datasets structured in the format described [here](dataformat.md).


### Example

From the project root directory, run the estimator with configuration specified by the option `-cfg cfg/vio.json` on the `data9_workbench` sequence (option `-seq data9_workbench`) which resides in directory `$DATAROOT` (option `-root $DATAROOT`) as the following:

```
bin/vio -cfg cfg/vio.json -root $DATAROOT -seq data9_workbench -out out_state -dataset xivo
```

where estimated states are saved to the output file `out_state`.

For detailed usage of the application, see the flags defined at the beginning of the application source file `src/app/vio.cpp`. 


### System logging

We use [glog](https://github.com/google/glog) for system logging. If you want to log runtime information to a text file for inspection later, first create a directory to hold logs

```
mkdir /YOUR/LOG/DIRECTORY/HERE
```

and then run the estimator with a prefix to specify the log directory:

```
GLOG_log_dir=/YOUR/LOG/DIRECTORY/HERE bin/vio -cfg cfg/vio.json -root $DATAROOT -seq data9_workbench -out out_state -dataset xivo
```

Note, by default, log is suppressed for runtime speed by setting `add_definitions(-DGOOGLE_STRIP_LOG=1)` in `src/CMakeLists.txt`. To enable log, simply comment out that line and re-build.

For more details on glog, see the tutorial [here](http://rpg.ifi.uzh.ch/docs/glog.html).

## ROS support

### Build

By default the ROS node of the system will not be built in case that your operating system does not have ROS installed. 

However, if you want to use the system with ROS support, you need to first install ROS (see instructions [here](http://wiki.ros.org/ROS/Installation)), and then turn on the `BUILD_ROSNODE` option when generating the makefile: In `build` directory of the project root, do the following:

```
cmake .. -DBUILD_ROSNODE=ON
```

followed by `make` to build with ROS support.

### Launch

1. In the project root directory, `source build/devel/setup.zsh`. If another shell, such as bash/sh, is used, source the corresponding shell script (`setup.bash`/`setup.sh`) respectively.
2. `roslaunch node/launch/xivo.launch` to launch the ros node, which subscribes to `/imu0` and `/cam0/image_raw` topics.
3. In antoher terminal, playback the rosbag of your choice, i.e., `rosbag play /PATH/TO/YOUR/ROSBAG`.

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

A simple python binding is provided by wrapping some public interfaces of the estimator via [pybind11](https://github.com/pybind/pybind11). Check out `pybind11/pyxivo.cpp` for the available interfaces in python. With pybind11, it is relatively easy if you want to expose more interfaces of the C++ implementation to python.

An example of using the Python binding is available in `scripts/pyxivo.py`, which demonstrates estimator creation, data loading, and visualization in python.

To run the demo, execute:

```
python scripts/pyxivo.py -cfg cfg/phab.json -root $DATAROOT -seq data9_workbench -dataset xivo -use_viewer
```

in the project root directory. The command-line options are more or less the same as the C++ executable. For detailed usage, you can look at the options defined at the beginning of the script `scripts/pyxivo.py`. Note you might need to install some python dependencies by executing the following in the project root directory:

```
pip install -r requirements.txt
```

<!-- ### Evaluation

We provide a python script `scripts/run_and_eval_pyxivo.py` to run the estimator on a specified TUM-VI sequence and benchmark the performance in terms ATE (Absolute Trajectory Error) and RPE (Relative Pose Error). To use it, execute the following in the project root directory:

```
python scripts/run_and_eval_pyxivo.py -root $TUMVIROOT -seq room6 -stdout -out_dir tmp -use_viewer
```
The `-seq` and `-root` options are the same as explained above. If the `-stdout` option is on, the script will print out the benchmarked performance to the terminal; `-out_dir` specifies the directory to save state estimates; `-use_viewer` option will turn on a 3D visualization. For detailed usage about the script, see the options defined at the beginning of the script. -->

## Evaluation

We benchmarked the performance of our system in terms of ATE (Absolute Trajectory Error), RPE (Relative Pose Error), and computational cost against other top-performing open-source implementations, i.e., OKVIS [Leutenegger *et al.*], VINS-Mono [Qin *et al.*] and ROVIO [Bloesch *et al.*], on publicly available datasets. 
*Our implementation achieves comparable accuracy at a fraction of the computational cost.* On a desktop PC equipped with an Intel Core i7 CPU @ 3.6 GHz, our system runs at around 140 Hz at low CPU consumption rate. As a comparison, OKVIS and VINS-Mono runs at around 20 Hz, and ROVIO runs at around 60 Hz. The runtime of our system can be further improved by better utilizing CPU cache and memory.

### Algorithm Categories

OKVIS and VINS-Mono are optimization based, which means they operate on keyframes in an iterative manner, which in general results in more accurate pose estimates at the price of higher latency and computational cost. ROVIO and XIVO are filtering based, which are causal and much cheaper in terms of computatioanl cost. Yet, they produce pose estimates comparable to optimization based methods.

Besides, OKVIS runs on stereo images, whereas the other three methods only use monocular images.

### Computational Cost

We benchmarked the runtime of OKVIS, VINS-Mono, ROVIO and XIVO on a desktop machine equipped with an Intel Core i7 CPU @ 3.6 GHz. The table below shows the runtime of the feature processing and state update modules.

| Module | OKVIS (Stereo+Keyframe) | VINS-Mono (Keyframe) | ROVIO | XIVO |
|:---       | :---   | :---       | :---   | :---  |
| Feature detection \& matching   | 15ms | 20ms | 1ms<sup>*</sup> | 3 ms|
| State update | 42ms | 50m | 13ms | 4 ms |


\* ROVIO is a 'direct' method that skips the feature matching step and directly uses the photometric error as the innovation term in EKF update step. Since it uses Iterative Extended Kalman Filter (IEKF) for state update, it's slower than our EKF-based method.

OKVIS and VINS-Mono (marked with Keyframe) perform iterative nonlinear least square on keyframes for state estimation, and thus are much slower in the state update step.

### Accuracy

We compared the performance of our system in terms of ATE and RPE on two publicly available datasets: TUM-VI and EuRoC. We achieve comparable pose estimation accuracy at a fraction of the computational cost of the top-performing open-source implementations.

#### TUM-VI

The following table shows the performance on 6 indoor sequences where ground-truth poses are available. The numbers for OKVIS, VINS-Mono, and ROVIO are taken from the TUM-VI benchmark paper. The evaluation script of XIVO can be found in `misc/run_all.sh`.

| Sequence | length | OKVIS (Stereo+Keyframe) | VINS-Mono (Keyframe) | ROVIO | XIVO |
|:---       | :---    | :---:   | :---:       | :---:   | :---:  |
|room1     | 156m   | **0.06m** | 0.07m | 0.16m | 0.13m |
|room2     | 142m   | 0.11m | **0.07m** | 0.33m | 0.11m |
|room3     | 135m   | **0.07m**  | 0.11m | 0.15m | 0.17m |
|room4     | 68m    | **0.03m** | 0.04m | 0.09m | 0.09m |
|room5     | 131m   | **0.07m** | 0.20m | 0.12m | 0.10m |
|room6     | 67m    | **0.04m** | 0.08m | 0.05m | 0.05m |

*Table 1. RMSE ATE* in meters. Methods marked with *Keyframe* are keyframe-based, others are recursive approaches.

| Sequence | OKVIS (Stereo+Keyframe) | VINS-Mono (Keyframe) | ROVIO | XIVO |
|:---       | :---:   | :---:       | :---:   | :---:  |
|room1 | **0.013**m/**0.43**<sup>o</sup> | 0.015m/0.44<sup>o</sup> | 0.029m/0.53<sup>o</sup> | 0.022m/0.60<sup>o</sup> |
|room2 | **0.015**m/**0.62**<sup>o</sup> | 0.017m/0.63<sup>o</sup> | 0.030m/0.67<sup>o</sup> | 0.040m/0.71<sup>o</sup> |
|room3 | **0.012**m/0.64<sup>o</sup> | 0.023m/**0.63**<sup>o</sup> | 0.027m/0.66<sup>o</sup> | 0.086m/0.74<sup>o</sup> |
|room4 | **0.012**m/0.57<sup>o</sup> | 0.015m/**0.41**<sup>o</sup> | 0.022m/0.61<sup>o</sup> | 0.022m/0.62<sup>o</sup> |
|room5 | **0.012**m/**0.47**<sup>o</sup> | 0.026m/**0.47**<sup>o</sup> | 0.031m/0.60<sup>o</sup> | 0.030m/0.60<sup>o</sup> |
|room6| **0.012**m/0.49<sup>o</sup> | 0.014m/**0.44**<sup>o</sup> | 0.019m/0.50<sup>o</sup> | 0.020m/0.52<sup>o</sup> |

*Table 2. RMSE RPE* in translation (meters) and rotation (degrees). Methods marked with *Keyframe* are keyframe-based, others are recursive approaches.

#### EuRoC

Benchmark results on the EuRoC dataset will be available soon.
