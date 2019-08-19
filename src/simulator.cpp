
#include <chrono>
#include <iostream>
#include <math.h>
#include <memory>
#include <random>

#include "glog/logging.h"
#include "opencv2/imgproc/imgproc.hpp"

#include "simulator.h"
#include "visualize.h"

namespace feh {

Simulator::Simulator(const std::string &cfg_path) {
  LOG(INFO) << "Initializing simulator... " << std::endl;
  cfg_ = LoadJson(cfg_path);
  total_time_ = cfg_["total_time"].asDouble(); // seconds
  imu_hz_ = cfg_["imu_hz"].asDouble();
  cam_hz_ = cfg_["camera_hz"].asDouble();
  total_count_ = static_cast<int>(ceil(imu_hz_ * total_time_));
  counter_ = 0;
  curr_sim_time_ = 0;
  frame_counter_ = 0;
  motion_type_ = 0; // easiest motion type
  init_period_ = cfg_["init_static_period"].asDouble();

  num_pts_to_track_ = cfg_["num_pts_to_track"].asUInt();
  min_num_pts_to_track_ =
      cfg_["min_num_pts_to_track"].asUInt(); // when number of tracks is below
                                             // this threshold, add more tracks

  // Measurement Noises
  gyro_noise_std_ = cfg_["gyro_noise_std"].asDouble();
  accel_noise_std_ = cfg_["accel_noise_std"].asDouble();
  tracking_noise_std_ = cfg_["tracking_noise_std"].asDouble();
  max_track_lifetime_ = cfg_["max_track_lifetime"].asInt();

  // camera sampling rate
  cam_sample_rate_ = imu_hz_ / cam_hz_;

  // State generation noises
  //    model_noise_da_ = 0.005; // for motion_type ?
  //    model_noise_dw_ = 0.005;

  // initialize camera model
  auto cam_cfg = cfg_["camera_cfg"].isString()
                     ? LoadJson(cfg_["camera_cfg"].asString())
                     : cfg_["camera_cfg"];
  cam_ = new Camera(cam_cfg);
  LOG(INFO) << "camera initialized" << std::endl;

  mask_ = cv::Mat(cam_->rows(), cam_->cols(), CV_8UC1);
  z_near_ = cfg_.get("z_nar", 0.05).asDouble();
  z_far_ = cfg_.get("z_far", 10.0).asDouble();

  z_dist_ = std::uniform_real_distribution<ftype>(z_near_, z_far_);
  x_dist_ = std::uniform_real_distribution<ftype>(0, cam_->cols());
  y_dist_ = std::uniform_real_distribution<ftype>(0, cam_->rows());

  drop_prob_ = cfg_["track_drop_probability"].asDouble();
  mask_size_ = cfg_["mask_size"].asDouble() / 2.0;
  motion_type_ = cfg_["motion_type"].asInt();
  delay_camera_time_ = cfg_["delay_camera_time"].asDouble();

  // measurement noise distributions
  if (!cfg_["fixed_seed"].asBool()) {
    long seed = std::chrono::system_clock::now().time_since_epoch().count();
    LOG(INFO) << "\nseed=" << seed << std::endl;
    gen_ = std::shared_ptr<std::knuth_b>(new std::knuth_b(seed));
  } else {
    long seed = cfg_["seed"].asInt64();
    LOG(INFO) << "\nuse saved seed=" << seed << std::endl;
    gen_ = std::shared_ptr<std::knuth_b>(new std::knuth_b(seed));
  }

  // Construct simulated trajectories
  translations_.clear();
  velocities_.clear();
  accelerations_.clear();
  GenerateTrajectory();

  // Initialize simulated state
  auto X = cfg_["X"];
  X_.Rsb = SO3::exp(GetVectorFromJson<ftype, 3>(X, "W"));
  X_.Tsb = GetVectorFromJson<ftype, 3>(X, "T");
  X_.Vsb = GetVectorFromJson<ftype, 3>(X, "V");
  X_.bg = GetVectorFromJson<ftype, 3>(X, "bg");
  X_.ba = GetVectorFromJson<ftype, 3>(X, "ba");
  X_.Rbc = SO3::exp(GetVectorFromJson<ftype, 3>(X, "Wbc"));
  X_.Tbc = GetVectorFromJson<ftype, 3>(X, "Tbc");
  Vec3 Wg;
  Wg.head<2>() = GetVectorFromJson<ftype, 2>(X, "Wg");
  X_.Rg = SO3::exp(Wg);

  // Initialize grav Vector WARNING: Make sure filter has the same one
  g_ << 0, 0, -9.8;
  // Initialize imu calibration
  Cas_.setIdentity();
  Car_.setIdentity();
  Cgs_.setIdentity();
  Cgr_.setIdentity();
  LOG(INFO) << "Simulator initialized";
}

bool Simulator::GetMeas(msg::IMU &imu_msg, msg::Track &track_msg) {
  ftype dt = 1.0 / imu_hz_;
  imu_msg.ts_ = curr_sim_time_;

  // both accel and gyro below are measured in spatial frame
  Vec3 accel = accelerations_[counter_];
  Vec3 gyro = rotational_velocities_[counter_];
  SO3 Rsb = X_.Rsb;

  // fix the the gravity mis-alignment (not really used here, since Rg=I)
  // and transfer to body frame
  SO3 Rg;
  accel = Rsb.inv() * (accel - X_.Rg * g_); // in body frame

  accel += X_.ba;
  imu_msg.accel_ = accel;
  gyro += X_.bg;
  imu_msg.gyro_ = gyro;

  // Propagate the true state
  Vec3 gyro_calib = Cgr_ * Cgs_ * (gyro - X_.bg);
  Vec3 accel_calib = Car_ * Cas_ * (accel - X_.ba);
  X_.Rsb = Rsb * SO3::exp(gyro_calib * dt);
  X_.Tsb += X_.Vsb * dt;
  X_.Vsb += (Rsb * accel_calib + X_.Rg * g_) * dt;

  // Add measurement noise
  imu_msg.accel_ += RandomVector<3>(0, accel_noise_std_, gen_);
  imu_msg.gyro_ += RandomVector<3>(0, gyro_noise_std_, gen_);
  imu_msg.initialized_ = true;

  // Generate visual measurement
  if (counter_ % cam_sample_rate_ == 0) {
    // Pull out the measurement for pose

    const auto &Rsb = X_.Rsb;
    const auto &Rbc = X_.Rbc;

    const auto &Tsb = X_.Tsb;
    const auto &Vsb = X_.Vsb;
    const auto &Tbc = X_.Tbc;

    // camera to spatial frame transformation
    SO3 Rsc = Rsb * Rbc;
    Vec3 Tsc = Rsb * Tbc + Tsb;

    SO3 Rcs = Rsc.inv();
    Vec3 Tcs = -(Rcs * Tsc);

    // generate random points in camera frame (always in the front of the
    // camera)
    // and bring them to the spatial frame

    // remove invalid tracks
    auto it = std::remove_if(pts_.begin(), pts_.end(), [](FeaturePtr f) {
      return f->status_ == FeatureStatus::REJECTED_BY_TRACKER ||
             f->status_ == FeatureStatus::REJECTED_BY_FILTER ||
             f->status_ == FeatureStatus::DROPPED;
    });
    pts_.erase(it, pts_.end());

    // clear mask
    mask_.setTo(cv::Scalar(1));
    std::vector<FeaturePtr> pts(pts_.begin(), pts_.end());
    std::sort(pts.begin(), pts.end(), [](FeaturePtr f1, FeaturePtr f2) -> bool {
      return f1->score_ > f2->score_;
    });
    pts_.resize(pts.size());
    std::copy(pts.begin(), pts.end(), pts_.begin());

    // "track" the features
    uint32_t valid_tracks = 0;
    int total_drop = 0;
    for (auto f : pts_) {
      Vec3 Xs = f->sim_.Xs;
      Vec3 Xc = Rcs * Xs + Tcs;
      if (Xc(2) <= z_near_ || Xc(2) > z_far_) {
        f->track_.status_ = TrackStatus::REJECTED;
        continue;
      }
      Vec2 xc = Xc.head<2>() / Xc(2);

      Vec2 xp = cam_->Project(xc);
      xp += RandomVector<2>(0.0, tracking_noise_std_, gen_);
      bool track_ok(true);

      if (OutOfView(xp, *cam_)) {
        f->track_.status_ = TrackStatus::DROPPED;
        track_ok = false;
        ++total_drop;
      } else {
        if (std::uniform_real_distribution<ftype>(0, 1)(*gen_) < drop_prob_) {
          f->track_.status_ = TrackStatus::DROPPED;
          track_ok = false;
          ++total_drop;
        } else if (f->track_.size() > f->sim_.lifetime) {
          // drop tracks exceeding max lifetime with high probability
          f->track_.status_ = TrackStatus::DROPPED;
          track_ok = false;
          ++total_drop;
        }
      }

      if (track_ok) {
        if (!mask_.at<uchar>(int(xp(1)), int(xp(0)))) {
          f->track_.status_ = TrackStatus::REJECTED;
          track_ok = false;
        } else {
          auto x(xp(0)), y(xp(1));
          cv::rectangle(mask_, cv::Point2d(x - mask_size_, y - mask_size_),
                        cv::Point2d(x + mask_size_, y + mask_size_),
                        cv::Scalar(0), -1);
        }

        ++valid_tracks;
        f->UpdateTrack(xp);
        f->track_.status_ = TrackStatus::TRACKED;
      }
    }
    VLOG(0) << "Total drop=" << total_drop << std::endl;
    // only add more tracks when number of valid tracks is below a threshold
    for (; valid_tracks < min_num_pts_to_track_; ++valid_tracks) {
      ftype z = z_dist_(*gen_);
      Vec2 xp(x_dist_(*gen_), y_dist_(*gen_));
      while (OutOfView(xp, *cam_) || !mask_.at<uchar>((int)xp[1], (int)xp[0])) {
        xp << x_dist_(*gen_), y_dist_(*gen_);
      }
      // mask out the occupied region
      auto x(xp(0)), y(xp(1));
      cv::rectangle(mask_, cv::Point2d(x - mask_size_, y - mask_size_),
                    cv::Point2d(x + mask_size_, y + mask_size_), cv::Scalar(0),
                    -1);
      Vec2 xc = cam_->UnProject(xp);
      auto xp_meas = xp + RandomVector<2>(0.0, tracking_noise_std_, gen_);

      FeaturePtr f = std::make_shared<Feature>(xp_meas(0), xp_meas(1));
      Vec3 Xs = Rsc * Vec3{xc(0) * z, xc(1) * z, z} + Tsc;
      // set ground truth
      f->sim_.Xs = Xs;
      f->sim_.xp = xp;
      f->sim_.xc = xc;
      f->sim_.z = z;
      f->sim_.lifetime =
          std::uniform_int_distribution<int>(2, max_track_lifetime_)(*gen_);

      // simulate score of a feature detector
      f->score_ = std::uniform_real_distribution<float>(0, 1)(*gen_);

      pts_.emplace_back(f);
    }

    track_msg.features_.resize(pts_.size());
    std::copy(pts_.begin(), pts_.end(), track_msg.features_.begin());
    track_msg.initialized_ = true;
    track_msg.ts_ = imu_msg.ts_;

    LOG(INFO) << "simulated features: " << pts_.size();

    // debug
    int mean_track_lifetime =
        std::accumulate(pts_.begin(), pts_.end(), 0, [](int x, FeaturePtr f) {
          return x + f->track_.size();
        });
    mean_track_lifetime /= pts_.size();
    LOG(INFO) << "mean track life=" << mean_track_lifetime;
  }

  ++counter_;
  curr_sim_time_ += dt;
  return counter_ < total_count_;
}

SO3 fixor(const Vec3 &A, const Vec3 &B) {
  Vec3 x(1, 0, 0);
  Vec3 y = A - B;
  Vec3 z = y / y.norm();

  y = z.cross(x);
  y /= y.norm();
  // now y is a unit vector perpendicular to x-z plane
  x = y.cross(z);
  x /= x.norm();
  // now x is a unit vector perpendicular to y-z plane

  // overall, direction of z is fixed to (B->A)
  // and the other directions are constructed properly
  Mat3 R;
  R.setZero();
  R.block<3, 1>(0, 0) = x;
  R.block<3, 1>(0, 1) = y;
  R.block<3, 1>(0, 2) = z;
  return SO3{R};
}

void Simulator::GenerateTrajectory() {
  LOG(INFO) << "Generating simulated trajectory type " << motion_type_
            << std::endl;
  ftype delta_t = 1.0 / imu_hz_; // TODO: make clock for sim work with differing
                                 // rates of imu and cam
  ftype time = 0;

  // parameters controling the synthetic data
  Vec3 rate, mult;

  // random frequencies and multipliers
  rate = RandomVector<3>(0., 1., gen_);
  mult = RandomVector<3>(0., 1., gen_);

  // saved set
  rate << 15, 30, 15;
  rate /= 180.0;
  mult << 2, 3, 4;
  Vec3 init_slope(mult / (init_period_ * init_period_));

  // generate the trajectory
  for (int i = 0; i < total_count_ + 2; ++i) {
    ftype u = time * time / (init_period_ + time);
    //        ftype u = time * std::sqrt(time) / (10 + time);
    //        ftype u = time;
    Vec3 radius(mult);
    if (time < init_period_) {
      radius -= init_slope * (time - init_period_) * (time - init_period_);
    }
    if (motion_type_ == 0) {
      translations_.emplace_back(radius[0] * std::cos((M_PI * rate[0] * u)) *
                                     std::sin((M_PI * rate[0]) * u),
                                 radius[1] * std::cos((M_PI * rate[1] * u)) *
                                     std::cos((M_PI * rate[1]) * u),
                                 radius[2] * std::sin((M_PI * rate[2]) * u));
    } else if (motion_type_ == 1) {
      translations_.emplace_back(radius[0] * std::sin((M_PI * rate[0]) * u),
                                 radius[1] * std::sin((M_PI * rate[1]) * u),
                                 radius[2] * std::cos((M_PI * rate[2]) * u));
    } else if (motion_type_ == 2) {
      // most challenging
      translations_.emplace_back(radius[0] * std::sin((M_PI * rate[0]) * u),
                                 radius[1] * std::sin((M_PI * rate[1]) * u),
                                 radius[2] *
                                     (1 - std::cos((M_PI * rate[2]) * u)));
    } else if (motion_type_ == 3) {
      translations_.emplace_back(5 * std::cos(M_PI * rate[0] * u),
                                 5 * std::sin(M_PI * rate[1] * u), 0);
    } else if (motion_type_ == 4) {
      // stationary
      translations_.emplace_back(0, 0, 0);
    } else {
      LOG(FATAL) << "only motion type 0, 1, 2, 3 are supported";
    }
    time += delta_t;
  }
  LOG(INFO) << "translation generated";

  // differentiate
  for (int i = 1; i < total_count_ + 2; ++i)
    velocities_.emplace_back((translations_[i] - translations_[i - 1]) /
                             delta_t);
  LOG(INFO) << "velocity generated";

  for (int i = 1; i < total_count_ + 1; ++i)
    accelerations_.emplace_back((velocities_[i] - velocities_[i - 1]) /
                                delta_t);
  accelerations_.push_back(accelerations_[total_count_ - 1]);
  LOG(INFO) << "acceleration generated";

  Vec3 look_at; // point to look at
  // look_at = RandomVector<3>(1, 2, gen_);
  look_at = Vec3(0, 0, 5);

  // get rotational velocities
  for (int i = 1; i < total_count_ + 2; ++i) {
    SO3 Ra, Rb;
    if (motion_type_ <= 2) {
      Rb = fixor(translations_[i], look_at);
      Ra = fixor(translations_[i - 1], look_at);
    } else if (motion_type_ == 3) {
      Rb = fixor(translations_[i], translations_[i] + Vec3(0, 0, 1.0));
      Ra = fixor(translations_[i - 1], translations_[i - 1] + Vec3(0, 0, 1.0));
    } else if (motion_type_ == 4) {
      // do nothing
    }
    auto rot_vel = SO3::log(Ra.inv() * Rb) / delta_t;
    rotational_velocities_.push_back(rot_vel);
  }
  LOG(INFO) << "orientation generated" << std::endl;
}

cv::Mat Simulator::VisualizeTracks() {
  cv::Mat disp(cam_->rows(), cam_->cols(), CV_8UC3);
  disp.setTo(cv::Scalar(0, 0, 0));
  return feh::VisualizeTracks(disp, pts_);
}
}
