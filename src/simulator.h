#pragma once

#include <fstream>
#include <iostream>
#include <istream>
#include <random>

#include "core.h"
#include "estimator.h"
#include "feature.h"

#undef SIMULATE_VISUAL_MEAS

namespace feh {

struct Simulator {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Simulator(const std::string &cfg_path);

  /// \brief get and pass measurements to motion integtration of simulator
  /// \param imu_dat: imu measurement with Gaussian noise
  bool GetMeas(msg::IMU &imu_msg, msg::Track &track_msg);
  /// \brief internal simulation generation functions
  void GenerateTrajectory();
  cv::Mat VisualizeTracks();

  Json::Value cfg_;

  ////////////////////////////////////////////////////////////////////////////////
  // Clone the following to initialize estimator in simulation
  ////////////////////////////////////////////////////////////////////////////////
  State X_;
  Eigen::Matrix<ftype, kFullSize, 1> err_; // error state
  Mat3 Cas_, Car_;
  Mat3 Cgs_, Cgr_;
  Vec3 g_;
  ////////////////////////////////////////////////////////////////////////////////

  std::list<FeaturePtr> pts_;
  Camera *cam_;

  cv::Mat mask_;
  uint32_t num_pts_to_track_;
  uint32_t min_num_pts_to_track_; // when number of tracks is below this
                                  // threshold, add more tracks
  ftype z_near_, z_far_;          // near and far plane
  std::uniform_real_distribution<ftype> x_dist_, y_dist_, z_dist_;

  ftype drop_prob_;
  ftype mask_size_;
  int motion_type_;
  ftype init_period_;

  // sampling rates and duration
  int64_t total_time_, curr_sim_time_; // seconds
  ftype imu_hz_, cam_hz_;
  int counter_, total_count_, cam_sample_rate_;
  int frame_counter_;

  ftype delay_camera_time_;

  // generation noises
  ftype accel_noise_std_, gyro_noise_std_;
  ftype tracking_noise_std_;
  int max_track_lifetime_;

  ftype model_noise_da_, model_noise_dw_;

  // trajectory generation
  std::vector<Vec3> translations_;
  std::vector<Vec3> velocities_;
  std::vector<Vec3> accelerations_;
  std::vector<Vec3> rotational_velocities_;

  // measurement noise generators
  std::shared_ptr<std::knuth_b> gen_;
};

using SimulatorPtr = std::shared_ptr<Simulator>;

} // feh
