#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <tuple>

#include "Eigen/QR"
#include "glog/logging.h"

#include "estimator.h"
#include "feature.h"
#include "group.h"
#include "jac.h"
#include "mm.h"
#include "param.h"
#include "tracker.h"
#include "helpers.h"
#include "mapper.h"

#ifdef USE_G2O
#include "optimizer.h"
#endif

namespace xivo {

std::unique_ptr<Estimator> Estimator::instance_{nullptr};

EstimatorPtr Estimator::Create(const Json::Value &cfg) {
  if (instance_) {
    LOG(WARNING) << "Estimator already exists!";
  } else {
    instance_ = std::unique_ptr<Estimator>(new Estimator{cfg});
  }
  return instance_.get();
}

EstimatorPtr Estimator::instance() {
  if (!instance_) {
    LOG(FATAL) << "Estimator NOT created yet!";
  }
  return instance_.get();
}

static const Mat3 I3{Mat3::Identity()};
static const Mat3 nI3{-I3};
static const Mat2 I2{Mat2::Identity()};
static const Mat2 nI2{-I2};

static bool cmp(const std::unique_ptr<internal::Message> &m1,
                const std::unique_ptr<internal::Message> &m2) {
  return m1->ts() > m2->ts();
}

namespace internal {
void Inertial::Execute(Estimator *est) {
  est->InertialMeasInternal(ts_, gyro_, accel_);
}

void Visual::Execute(Estimator *est) { est->VisualMeasInternal(ts_, img_); }

void VisualTrackerOnly::Execute(Estimator *est) { est->VisualMeasInternalTrackerOnly(ts_, img_); }

void VisualPointCloud::Execute(Estimator *est) {
  est->VisualMeasPointCloudInternal(ts_, feature_ids_, xp_vals_);
}

void VisualPointCloudTrackerOnly::Execute(Estimator *est) {
  est->VisualMeasPointCloudInternalTrackerOnly(ts_, feature_ids_, xp_vals_);
}

} // namespace internal

// destructor
Estimator::~Estimator() {
  if (cfg_.get("print_calibration", false).asBool()) {
    std::cout << "===== Auto-Calibration =====\n";
    std::cout << "Rbc=\n" << X_.Rbc.matrix() << std::endl;
    auto Wbc = X_.Rbc.log();
    std::cout << "Wbc=" << Wbc.transpose() << std::endl;
    std::cout << "Tbc=" << X_.Tbc.transpose() << std::endl;
    std::cout << "td=" << X_.td << std::endl;
    std::cout << "gyro.bias=" << X_.bg.transpose() << std::endl;
    std::cout << "accel.bias=" << X_.ba.transpose() << std::endl;
    std::cout << "Rsg=" << X_.Rsg.matrix() << std::endl;
    auto Wsg = X_.Rsg.log();
    std::cout << "Wsg=" << Wsg.transpose() << std::endl;
    std::cout << "===== IMU intrinsics =====\n";
    std::cout << "Ca=\n" << imu_.Ca() << std::endl;
    std::cout << "Cg=\n" << imu_.Cg() << std::endl;
    std::cout << "===== Camera intrinsics =====\n";
    CameraManager::instance()->Print(std::cout);
  }

  if (worker_) {
    worker_->join();
    delete worker_;
  }
}

Estimator::Estimator(const Json::Value &cfg)
    : cfg_{cfg}, gauge_group_{-1}, worker_{nullptr}, timer_{"estimator"},
      gauge_group_ptr_{nullptr} {

  // /////////////////////////////
  // Component flags
  // /////////////////////////////
  simulation_ = cfg_.get("simulation", false).asBool();
  use_canvas_ = cfg_.get("use_canvas", true).asBool();
  print_timing_ = cfg_.get("print_timing", false).asBool();
  integration_method_ =
      cfg_.get("integration_method", "unspecified").asString();

  // OOS update options
  use_OOS_ = cfg_.get("use_OOS", false).asBool();
  use_compression_ = cfg_.get("use_compression", false).asBool();
  compression_trigger_ratio_ =
      cfg_.get("compression_trigger_ratio", 1.5).asDouble();
  OOS_update_min_observations_ =
      cfg_.get("OOS_update_min_observations", 5).asInt();

  // IMU clamping
  Vec3 _vec_;
  clamp_signals_ = cfg_.get("clamp_signals", false).asBool();
  max_accel_ = GetVectorFromJson<number_t, 3>(cfg_, "max_accel");
  max_gyro_ = GetVectorFromJson<number_t, 3>(cfg_, "max_gyro");

   // one point ransac parameters
  use_1pt_RANSAC_ = cfg_.get("use_1pt_RANSAC", false).asBool();
  ransac_thresh_ = cfg_.get("1pt_RANSAC_thresh", 5).asDouble();
  ransac_prob_ = cfg_.get("1pt_RANSAC_prob", 0.95).asDouble();
  ransac_Chi2_ = cfg_.get("1pt_RANSAC_Chi2", 5.89).asDouble();

  // depth-initialization subfilter options
  number_t tri_std = cfg_["subfilter"].get("visual_meas_std", 3.5).asDouble();
  subfilter_options_.Rtri = tri_std * tri_std;
  subfilter_options_.MH_thresh =
      cfg_["subfilter"].get("MH_thresh", 5.991).asDouble();
  subfilter_options_.ready_steps =
      cfg_["subfilter"].get("ready_steps", 5).asInt();

  // depth optimization options
  use_depth_opt_ = cfg_.get("use_depth_opt", false).asBool();
  refinement_options_.two_view =
      cfg_["depth_opt"].get("two_view", false).asBool();
  refinement_options_.use_hessian = cfg_["depth_opt"].get("use_hessian", false).asBool();
  refinement_options_.max_iters = cfg_["depth_opt"].get("max_iters", 5).asInt();
  refinement_options_.eps = cfg_["depth_opt"].get("eps", 1e-4).asDouble();
  refinement_options_.damping =
      cfg_["depth_opt"].get("damping", 1e-3).asDouble();
  refinement_options_.max_res_norm =
      cfg_["depth_opt"].get("max_res_norm", 2.0).asDouble();
  refinement_options_.Rtri = subfilter_options_.Rtri;

  triangulate_pre_subfilter_ =
      cfg_.get("triangulate_pre_subfilter", false).asBool();
  triangulate_options_.method = cfg_["triangulation"].get("method", "l1_angular").asString();
  triangulate_options_.zmin =
      cfg_["triangulation"].get("zmin", 0.05).asDouble();
  triangulate_options_.zmax = cfg_["triangulation"].get("zmax", 5.0).asDouble();
  triangulate_options_.max_theta_thresh = cfg_["triangulation"].get("max_theta_thresh", 0.1).asDouble() * M_PI / 180;
  triangulate_options_.beta_thesh = cfg_["triangulation"].get("beta_thesh", 0.25).asDouble() * M_PI / 180;

  adaptive_initial_depth_options_.median_weight =
    cfg_["adaptive_initial_depth"].get("median_weight", 0.99).asDouble();
  adaptive_initial_depth_options_.min_feature_lifetime =
    cfg_["adaptive_initial_depth"].get("minimum_feature_lifetime", 5).asInt();

  remove_outlier_counter_ = cfg_.get("remove_outlier_counter", 10).asInt();

  group_degrees_fixed_ = cfg_.get("group_degrees_fixed", 4).asInt();
  if ((group_degrees_fixed_ != 4) && (group_degrees_fixed_ != 6)) {
    LOG(FATAL) << "group_degrees_fixed must be 4 or 6";
  }

  // load imu calibration
  auto imu_calib = cfg_["imu_calib"];
  // load accel axis misalignment first as a 3x3 matrix
  Mat3 Ta =
      GetMatrixFromJson<number_t, 3, 3>(imu_calib, "Car", JsonMatLayout::RowMajor);
  Mat3 Ka;  // accel scaling
  Ka.diagonal() = GetVectorFromJson<number_t, 3>(imu_calib, "Cas");
  Mat3 Ca{Ta * Ka};
  // load gyro axis misalignment first as 3x3 matrix
  Mat3 Tg =
      GetMatrixFromJson<number_t, 3, 3>(imu_calib, "Cgr", JsonMatLayout::RowMajor);
  Mat3 Kg;  // gyro scaling
  Kg.diagonal() = GetVectorFromJson<number_t, 3>(imu_calib, "Cgs");
  Mat3 Cg{Tg * Kg};
  // now update the IMU component
  imu_ = IMU{Ca, Cg};
  LOG(INFO) << "Imu calibration loaded";

  g_ = GetMatrixFromJson<number_t, 3, 1>(cfg_, "gravity");
  LOG(INFO) << "gravity loaded:" << g_.transpose();

  // /////////////////////////////
  // Initialize motion state
  // /////////////////////////////
  auto X = cfg_["X"];
  try {
    X_.Rsb = SO3::exp(GetVectorFromJson<number_t, 3>(X, "Wsb"));
  } catch (const Json::LogicError &e) {
    Mat3 Rsb_tmp = GetMatrixFromJson<number_t, 3, 3>(X, "Wsb", JsonMatLayout::RowMajor);
    if (Sophus::isOrthogonal(Rsb_tmp) && (Rsb_tmp.determinant() > 0.0)) {
      X_.Rsb = SO3(Rsb_tmp);
    } else {
      LOG(WARNING) << "Input value of Rsb is not orthogonal or its determinant is negative. Projecting to SO(3) group";
      X_.Rsb = SO3::fitToSO3(Rsb_tmp);
    }
  }
  X_.Tsb = GetVectorFromJson<number_t, 3>(X, "Tsb");
  X_.Vsb = GetVectorFromJson<number_t, 3>(X, "Vsb");
  X_.bg = GetVectorFromJson<number_t, 3>(X, "bg");
  X_.ba = GetVectorFromJson<number_t, 3>(X, "ba");

  if (cfg_.get("imu_tk_convention", false).asBool()) {
    // For biases obtained by IMU-TK library,
    // the calibrated meaurement is a_calib = K(a_raw + a_bias)
    // whereas in our model a_calib=K * a_raw - a_bias
    // thus we need convert that.
    X_.bg = -imu_.Cg() * X_.bg;
    X_.ba = -imu_.Ca() * X_.ba;
  }

  try {
    X_.Rbc = SO3::exp(GetVectorFromJson<number_t, 3>(X, "Wbc"));
  } catch (const Json::LogicError &e) {
    Mat3 Rbc_tmp = GetMatrixFromJson<number_t, 3, 3>(X, "Wbc", JsonMatLayout::RowMajor);
    if (Sophus::isOrthogonal(Rbc_tmp) && (Rbc_tmp.determinant() > 0.0)) {
      X_.Rbc = SO3(Rbc_tmp);
    } else {
      LOG(WARNING) << "Input value of Rbc is not orthogonal or its determinant is negative. Projecting to SO(3) group";
      X_.Rbc = SO3::fitToSO3(Rbc_tmp);
    }
  }
  X_.Tbc = GetVectorFromJson<number_t, 3>(X, "Tbc");
  Vec3 Wsg;
  Wsg.head<2>() = GetVectorFromJson<number_t, 2>(X, "Wsg");
  X_.Rsg = SO3::exp(Wsg);
// temporal offset
#ifdef USE_ONLINE_TEMPORAL_CALIB
  X_.td = X["td"].asDouble();
#endif

  // initialize error state
  err_.resize(kFullSize);
  err_.setZero();
  // make all group & feature slots available
  std::fill(gsel_.begin(), gsel_.end(), false);
  std::fill(fsel_.begin(), fsel_.end(), false);
  LOG(INFO) << "Initial state loaded";
  LOG(INFO) << X_;

  auto P = cfg_["P"];
  P_.setIdentity(kFullSize, kFullSize);
  P_.block<3, 3>(Index::Wsb, Index::Wsb) *= P["Wsb"].asDouble();
  P_.block<3, 3>(Index::Tsb, Index::Tsb) *= P["Tsb"].asDouble();
  P_.block<3, 3>(Index::Vsb, Index::Vsb) *= P["Vsb"].asDouble();
  P_.block<3, 3>(Index::bg, Index::bg) *= P["bg"].asDouble();
  P_.block<3, 3>(Index::ba, Index::ba) *= P["ba"].asDouble();
  P_.block<3, 3>(Index::Wbc, Index::Wbc) *= P["Wbc"].asDouble();
  try {
    P_.block<3, 3>(Index::Tbc, Index::Tbc) *= P["Tbc"].asDouble();
  } catch (const std::exception&) {
    auto Cov = GetVectorFromJson<number_t, 3>(P, "Tbc");
    P_.block<3, 3>(Index::Tbc, Index::Tbc) *= Cov.asDiagonal();
  }
  P_.block<2, 2>(Index::Wsg, Index::Wsg) *= P["Wsg"].asDouble();
#ifdef USE_ONLINE_TEMPORAL_CALIB
  P_(Index::td, Index::td) *= P["td"].asDouble();
#endif

#ifdef USE_ONLINE_IMU_CALIB
  // online IMU calibration
  P_.block<9, 9>(Index::Cg, Index::Cg) *= P["Cg"].asDouble();
  P_.block<6, 6>(Index::Ca, Index::Ca) *= P["Ca"].asDouble();
#endif
// online camera intrinsics calibration
// initialize covariance for camera intrinsics
#ifdef USE_ONLINE_CAMERA_CALIB
  int dim = Camera::instance()->dim();
  try {
    // homogeneous focal length and principal point error
    P_.block(kCameraBegin, kCameraBegin, 4, 4) *= P["FC"].asDouble();
  } catch (const std::exception &) {
    // non-homogeneous focal length and principal point error
    auto fc_var = GetVectorFromJson<number_t, 2>(P, "FC");
    P_.block(kCameraBegin, kCameraBegin, 2, 2) *= fc_var[0];
    P_.block(kCameraBegin+2, kCameraBegin+2, 2, 2) *= fc_var[1];
  }
  P_.block(kCameraBegin + 4, kCameraBegin + 4, dim - 4, dim - 4) *=
      P["distortion"].asDouble();
  P_.block(kCameraBegin + dim, kCameraBegin + dim, kMaxCameraIntrinsics - dim,
           kMaxCameraIntrinsics - dim) *= 0;
#endif
  // standard deviation -> covariance
  // P_.block<kMotionSize, kMotionSize>(0, 0) *=
  //     P_.block<kMotionSize, kMotionSize>(0, 0);
  P_ *= P_;

  LOG(INFO) << "Initial covariance loaded";

  // allocate spaces for Jacobians
  F_.resize(kMotionSize, kMotionSize);
  F_.setIdentity();
  G_.resize(kMotionSize, 12);
  G_.setZero();

  auto Qmodel = cfg_["Qmodel"];
  Qmodel_.setZero(kMotionSize, kMotionSize);
  Qmodel_.block<3, 3>(Index::Wsb, Index::Wsb) = I3 * Qmodel["Wsb"].asDouble();
  Qmodel_.block<3, 3>(Index::Wbc, Index::Wbc) = I3 * Qmodel["Wbc"].asDouble();
  Qmodel_.block<2, 2>(Index::Wsg, Index::Wsg) = I2 * Qmodel["Wsg"].asDouble();
  Qmodel_.block<kMotionSize, kMotionSize>(0, 0) *=
      Qmodel_.block<kMotionSize, kMotionSize>(0, 0);
  LOG(INFO) << "Covariance of process noises loaded";

  // /////////////////////////////
  // Initialize measurement noise
  // /////////////////////////////
  auto Qimu = cfg_["Qimu"];
  Qimu_.setIdentity(12, 12);
  Qimu_.block<3, 3>(0, 0) *= GetVectorFromJson<number_t, 3>(Qimu, "gyro").asDiagonal();
  Qimu_.block<3, 3>(3, 3) *= GetVectorFromJson<number_t, 3>(Qimu, "accel").asDiagonal();
  Qimu_.block<3, 3>(6, 6) *= GetVectorFromJson<number_t, 3>(Qimu, "gyro_bias").asDiagonal();
  Qimu_.block<3, 3>(9, 9) *= GetVectorFromJson<number_t, 3>(Qimu, "accel_bias").asDiagonal();
  Qimu_ *= Qimu_;
  LOG(INFO) << "Covariance of IMU measurement noise loaded";


  R_ = cfg_["visual_meas_std"].asDouble();
  R_ *= R_;

  Roos_ = cfg["oos_meas_std"].asDouble();
  Roos_ *= Roos_;

  Rlc_ = cfg["loop_closure_meas_std"].asDouble();
  Rlc_ *= Rlc_;

  LOG(INFO) << "R=" << R_ << "; Roos=" << Roos_ << "; Rlc=" << Rlc_;

  // /////////////////////////////
  // Load initial std on feature state
  // /////////////////////////////
  init_z_ = cfg_["initial_z"].asDouble();
  init_std_x_ = cfg_["initial_std_x"].asDouble();
  init_std_y_ = cfg_["initial_std_y"].asDouble();
  init_std_x_ /= Camera::instance()->GetFocalLength();
  init_std_y_ /= Camera::instance()->GetFocalLength();
  init_std_z_ = cfg_["initial_std_z"].asDouble();
  min_z_ = cfg_["min_depth"].asDouble();
  max_z_ = cfg_["max_depth"].asDouble();
  init_std_x_badtri_ = cfg_["initial_std_x_badtri"].asDouble();
  init_std_y_badtri_ = cfg_["initial_std_y_badtri"].asDouble();
  init_std_z_badtri_ = cfg_["initial_std_z_badtri"].asDouble();
  LOG(INFO) << "Initial covariance for features loaded";

  MeasurementUpdateInitialized_ = false;

  // /////////////////////////////
  // Outlier rejection options
  // /////////////////////////////
  use_MH_gating_ = cfg_.get("use_MH_gating", true).asBool();
  min_required_inliers_ = cfg_.get("min_inliers", 5).asInt();
  MH_thresh_ = cfg_.get("MH_thresh", 5.991).asDouble();
  MH_thresh_multipler_ = cfg_.get("MH_adjust_factor", 1.1).asDouble();
  // FIXME (xfei): used in HuberOnInnovation, but kinda overlaps with MH gating
  outlier_thresh_ = cfg_.get("outlier_thresh", 1.1).asDouble();
  feature_owner_change_cov_factor_ =
    cfg_.get("filter_owner_change_cov_factor", 1.5).asDouble();
  strict_criteria_timesteps_ = cfg_.get("strict_criteria_timesteps", 5).asInt();

  // Feature Gauge Options (just checks)
  int num_gauge_xy_features = cfg_.get("num_gauge_xy_features", 3).asInt();
  number_t collinear_cross_prod_thresh =
    cfg_.get("collinear_cross_prod_thresh", 1e-3).asDouble();
  if ((num_gauge_xy_features < 0) || (num_gauge_xy_features > 3)) {
    LOG(FATAL) << "Number of XY Gauge Features must be between 0 and 3";
  }

  // reset initialization status
  if (simulation_) {
    gravity_init_counter_ = 0;
    gravity_initialized_ = true;
  } else {
    gravity_init_counter_ = cfg_.get("gravity_init_counter", 20).asInt();
    gravity_initialized_ = false;
  }
  vision_initialized_ = false;
  // reset measurement counter
  imu_counter_ = 0;
  vision_counter_ = 0;
  // reset various timestamps
  last_imu_time_ = timestamp_t::zero();
  curr_imu_time_ = timestamp_t::zero();

  last_vision_time_ = timestamp_t::zero();
  curr_vision_time_ = timestamp_t::zero();

  last_time_ = timestamp_t::zero();
  curr_time_ = timestamp_t::zero();

  // random number generator
  rng_ = std::unique_ptr<std::default_random_engine>(
      new std::default_random_engine);

  async_run_ = cfg_.get("async_run", false).asBool();
  if (async_run_) {
    Run();
  }
}

void Estimator::Run() {
  worker_ = new std::thread([this]() {
    for (;;) {
      std::unique_ptr<internal::Message> msg;
      {
        std::scoped_lock lck(buf_.mtx);
        if (buf_.initialized && buf_.size() > InternalBuffer::MAX_SIZE) {
          msg = std::move(buf_.front());
          std::pop_heap(buf_.begin(), buf_.end(), cmp);
          buf_.pop_back();
        }
      }
      if (msg != nullptr) {
        // std::cout << "executing\n";
        msg->Execute(this);
      }
    }
  });
}

bool Estimator::InitializeGravity() {
  VLOG(0) << "attempt to initialize gravity";
  if (!simulation_) {
    if (gravity_init_buf_.size() < gravity_init_counter_) {
      return false;
    }
    VLOG(0) << "initializing gravity";

    // got enough stationary samples, estimate gravity
    Vec3 mean_accel = std::accumulate(gravity_init_buf_.begin(),
                                      gravity_init_buf_.end(), Vec3{0, 0, 0});
    mean_accel /= gravity_init_buf_.size();

    Vec3 accel_calib = imu_.Ca() * mean_accel - X_.ba;

    // FromTwoVectors(a, b): returns R such that b=R*a
    // we need R * accel + Rg * g_ == 0
    // And R = Identity
    // so accel = Rg * (-g_)
    Eigen::AngleAxis<number_t> AAg(
        Eigen::Quaternion<number_t>::FromTwoVectors(-g_, accel_calib));
    Vec3 Wsg(AAg.axis() * AAg.angle());
    Wsg(2) = 0;
    X_.Rsg = SO3::exp(Wsg);

    LOG(INFO) << "===== Wsg initialization =====";
    LOG(INFO) << "stationary accel samples=" << gravity_init_buf_.size();
    LOG(INFO) << "accel " << accel_calib.transpose();
    LOG(INFO) << "Wsg=" << Wsg.transpose();
    LOG(INFO) << "g=" << g_.transpose();
    LOG(INFO) << "The norm below should be small";
    LOG(INFO) << "|Rsb*a+Rg*g|=" << (X_.Rsb * accel_calib + X_.Rsg * g_).norm();
  }
  return true;
}

void Estimator::InertialMeasInternal(const timestamp_t &ts, const Vec3 &gyro,
                                     const Vec3 &accel) {
  if (!GoodTimestamp(ts))
    return;

  ++imu_counter_;

  Vec3 gyro_new;
  Vec3 accel_new;

  Vec3 grav_s = X_.Rsg * g_;
  Vec3 grav_b = X_.Rsb.inverse() * grav_s;

  if (clamp_signals_) {

    Vec3 accel_wout_grav = accel + grav_b;

    for (int i=0; i < 3; i++) {
      number_t sign_gyro = (gyro(i) > 0) ? 1.0 : -1.0;
      number_t sign_accel = (accel_wout_grav(i) > 0) ? 1.0 : -1.0;

      number_t gyro_mag = (abs(gyro(i)) > max_gyro_(i)) ? max_gyro_(i) : abs(gyro(i));
      number_t accel_mag = (abs(accel_wout_grav(i)) > max_accel_(i)) ? max_accel_(i) : abs(accel_wout_grav(i));

      gyro_new(i) = sign_gyro * gyro_mag;
      accel_new(i) = sign_accel * accel_mag;
    }
    accel_new -= grav_b;

  } else{
    gyro_new = gyro;
    accel_new = accel;
  }

  // initialize imu -- basically gravity
  if (!gravity_initialized_) {
    gravity_init_buf_.emplace_back(accel_new);

    if (InitializeGravity()) {
      curr_imu_time_ = last_time_ = ts;

      curr_accel_ = last_accel_ = accel_new;
      curr_gyro_ = last_gyro_ = gyro_new;

      gravity_initialized_ = true;
      gravity_init_buf_.clear();
      LOG(INFO) << "IMU initialized";
    }
  } else {
    // process inertials only after vision module initialized
    if (vision_initialized_) {
      last_time_ = curr_time_;
      curr_time_ = ts;

      curr_accel_ = accel_new;
      curr_gyro_ = gyro_new;

      last_imu_time_ = curr_imu_time_;
      curr_imu_time_ = ts;
      Propagate(false);
    }
  }
}

void Estimator::Propagate(bool visual_meas) {
#ifndef NDEBUG
  CHECK(gravity_initialized_)
      << "state progagation with un-initialized imu module";
#endif

  timer_.Tick("propagation");

  number_t dt;
  Vec3 accel0, gyro0; // initial condition for integration

  dt = std::chrono::duration<number_t>(curr_time_ - last_time_).count();
  if (dt == 0) {
    if (!simulation_) {
      LOG(WARNING) << "measurement timestamps coincide?";
    }
    return;
  }

  if (!visual_meas) {
    // this is an imu meas
    slope_accel_ = (curr_accel_ - last_accel_) / dt;
    slope_gyro_ = (curr_gyro_ - last_gyro_) / dt;

    accel0 = last_accel_;
    gyro0 = last_gyro_;

    last_accel_ = curr_accel_;
    last_gyro_ = curr_gyro_;
  } else {
    // this is a visual meas
    accel0 = last_accel_;
    gyro0 = last_gyro_;

    last_accel_ = accel0 + slope_accel_ * dt;
    last_gyro_ = gyro0 + slope_gyro_ * dt;
  }

  if (dt > 0.030) {
    LOG(WARNING) << "dt=" << dt << "  > 30 ms";
  }
  if (integration_method_ == "PrinceDormand") {
    PrinceDormand(gyro0, accel0, dt);
  } else if (integration_method_ == "Fehlberg") {
    Fehlberg(gyro0, accel0, dt);
  } else if (integration_method_ == "RK4") {
    RK4(gyro0, accel0, dt);
  } else {
    LOG(FATAL) << "Unknown integration method";
  }

  P_.block<kMotionSize, kMotionSize>(0, 0).noalias() += Qmodel_;
  timer_.Tock("propagation");
}

void Estimator::Fehlberg(const Vec3 &gyro0, const Vec3 &accel0, number_t dt) {
  throw NotImplemented();
}

void Estimator::ComposeMotion(State &X, const Vec3 &V,
                              const Eigen::Matrix<number_t, 6, 1> &gyro_accel,
                              number_t dt) {
  Vec3 gyro = gyro_accel.head<3>();
  Vec3 accel = gyro_accel.tail<3>();

  Vec3 gyro_calib = imu_.Cg() * gyro - X.bg;
  Vec3 accel_calib = imu_.Ca() * accel - X.ba;

  // integrate the nominal state
  X.Tsb += V * dt; //+ 0.5 * a * dt * dt;
  X.Vsb += (X.Rsb * accel_calib + X.Rsg * g_) * dt;
  X.Rsb *= SO3::exp(gyro_calib * dt);

  X.Rsb = SO3::fitToSO3(X.Rsb.matrix());
}

void Estimator::ComputeMotionJacobianAt(
    const State &X, const Eigen::Matrix<number_t, 6, 1> &gyro_accel) {

  Vec3 gyro = gyro_accel.head<3>();
  Vec3 accel = gyro_accel.tail<3>();

  Vec3 gyro_calib = imu_.Cg() * gyro - X.bg;   // \hat\omega in the doc
  Vec3 accel_calib = imu_.Ca() * accel - X.ba; // \hat\alpha in the doc

  // jacobian w.r.t. error state
  Mat3 Rsb = X.Rsb.matrix();
  Mat3 Rsg = X.Rsg.matrix();

  Eigen::Matrix<number_t, 3, 9> dWsb_dCg;
  for (int i = 0; i < 3; ++i) {
    // NOTE: use the raw measurement (gyro) here. NOT the calibrated one
    // (gyro_calib)!!!
    dWsb_dCg.block<1, 3>(i, 3 * i) = gyro;
  }

  Mat3 dV_dCaA = dAB_dB<3, 1>(Rsb);
  Eigen::Matrix<number_t, 3, 9> dCaA_dCa = dAB_dA<3, 3>(accel);
  Eigen::Matrix<number_t, 9, 6> dCa_dCau = dA_dAu<number_t, 3>();
  Eigen::Matrix<number_t, 3, 6> dV_dCau = dV_dCaA * dCaA_dCa * dCa_dCau;

  Mat3 dWsb_dWsb = -SO3::hat(gyro_calib);
  // static Mat3 dW_dbg = -I3;

  // static Mat3 dT_dV = I3;

  Mat3 dV_dWsb = -Rsb * SO3::hat(accel_calib);
  Mat3 dV_dba = -Rsb;

  Mat3 dV_dWsg = -Rsg * SO3::hat(g_); // effective dimension: 3x2, since Wg is 2-dim
  // Mat2 dWg_dWg = Mat2::Identity();


  Mat39 dWsb_dRsb;
  invrodrigues(Rsb, &dWsb_dRsb);


  F_.setZero(); // wipe out the delta added to F in the previous step

  for (int j = 0; j < 3; ++j) {
    F_.coeffRef(Index::Wsb + j, Index::bg + j) = -1;  // dW_dbg
    F_.coeffRef(Index::Tsb + j, Index::Vsb + j) = 1;  // dT_dV

    for (int i = 0; i < 3; ++i) {
      // W
      F_.coeffRef(Index::Wsb + i, Index::Wsb + j) = dWsb_dWsb(i, j);
      // F_.coeffRef(Index::W + i, Index::bg + j) = dW_dbg(i, j);
      // T
      // F_.coeffRef(Index::T + i, Index::V + j) = dT_dV(i, j);

      // V
      F_.coeffRef(Index::Vsb + i, Index::Wsb + j) = dV_dWsb(i, j);
      F_.coeffRef(Index::Vsb + i, Index::ba + j) = dV_dba(i, j);

      if (j < 2) {
        // NOTE: Wg is 2-dim, i.e., NO z-component
        F_.coeffRef(Index::Vsb + i, Index::Wsg + j) = dV_dWsg(i, j);
      }
    }
  }

#ifdef USE_ONLINE_IMU_CALIB
  for (int j = 0; j < 9; ++j) {
    for (int i = 0 ; i < 3; ++i) {
      F_.coeffRef(Index::Wsb + i, Index::Cg + j) = dWsb_dCg(i, j);
    }
  }
  for (int j = 0; j < 6; ++j) {
    for (int i = 0; i < 3; ++i) {
      F_.coeffRef(Index::Vsb + i, Index::Ca + j) = dV_dCau(i, j);
    }
  }
#endif

  // Mat3 dW_dng = -I3;
  // Mat3 dV_dna = -R;
  // Mat3 dbg_dnbg = I3;
  // Mat3 dba_dnba = I3;

  // jacobian w.r.t. noise
  G_.setZero();
  for (int i = 0; i < 3; ++i) {

    //G_.coeffRef(Index::Wsb + i, i) = -1;  // dWsb_dng
    G_.coeffRef(Index::bg + i, 6 + i) = 1;  // dbg_dnbg
    G_.coeffRef(Index::ba + i, 9 + i) = 1;  // dba_dnba

    for (int j = 0; j < 3; ++j) {
      G_.coeffRef(Index::Vsb + i, 3 + j) = -Rsb(i, j);  // dV_dna
    }

    for (int j = 0; j < 3; j++) {
      Mat3 dhatj = unstack( dhat<number_t>().col(j) );
      Mat3 R_dhatj = Rsb * dhatj;
      Vec9 minus_R_dhatj_vec = Eigen::Map<Vec9> (R_dhatj.transpose().data()) * -1.0;
      G_.coeffRef(Index::Wsb+i, j) = dWsb_dRsb.row(i) * minus_R_dhatj_vec; // dW_dng
    }
  }
}

bool Estimator::GoodTimestamp(const timestamp_t &now) {
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now);
  auto curr_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(curr_time_);
  if (now_ms < curr_ms) {
    LOG(WARNING) << StrFormat("now=%ld ms < curr=%ld ms", now_ms.count(),
                                    curr_ms.count());
    return false;
  } else {
    return true;
  }
}

void Estimator::UpdateSystemClock(const timestamp_t &now) {
  if (!vision_initialized_) {
    if (gravity_initialized_) {
      // only initialize vision module after gravity initialized
      curr_time_ = now;
      last_vision_time_ = curr_vision_time_;
      curr_vision_time_ = now;

      vision_initialized_ = true;
      LOG(INFO) << "vision initialized";
    }
  } else {
    last_time_ = curr_time_;
    curr_time_ = now;

    last_vision_time_ = curr_vision_time_;
    curr_vision_time_ = now;
  }
}

void Estimator::RemoveGroupFromState(GroupPtr g) {
#ifndef NDEBUG
  CHECK(g->instate()) << "free a group not instate";
  CHECK(g->sind() != -1) << "invalid state index";
  CHECK(gsel_[g->sind()]) << "Group not in state?!";
#endif

  VLOG(0) << "removing group #" << g->id();
  // change the covariance and error state
  int index = g->sind();

  gsel_[index] = false;
  g->SetSind(-1);
  g->SetStatus(GroupStatus::FLOATING);

  int offset = kGroupBegin + 6 * index;
  int size = err_.rows();

  err_.segment<6>(offset).setZero();
  P_.block(offset, 0, 6, size).setZero();
  P_.block(0, offset, size, 6).setZero();
}

void Estimator::RemoveFeatureFromState(FeaturePtr f) {

#ifndef NDEBUG
  CHECK((f->instate() && (f->track_status() == TrackStatus::REJECTED ||
                          f->track_status() == TrackStatus::DROPPED)) ||
        f->status() == FeatureStatus::REJECTED_BY_FILTER);
  CHECK(f->sind() != -1) << "invalid state index";
  CHECK(fsel_[f->sind()]) << "Feature not in state?!";
#endif

  VLOG(0) << "removing feature #" << f->id();
  int index = f->sind();

  fsel_[index] = false;
  f->SetSind(-1);

  int offset = kFeatureBegin + 3 * index;
  int size = err_.rows();

  err_.segment<3>(offset).setZero();
  P_.block(offset, 0, 3, size).setZero();
  P_.block(0, offset, size, 3).setZero();
}

void Estimator::AddGroupToState(GroupPtr g) {
#ifndef NDEBUG
  CHECK(!g->instate()) << "group already in state";
  CHECK(g->sind() == -1) << "group slot already allocated";
#endif

  // change the covariance and error state
  int index;
  // find empty slot
  for (index = 0; index < gsel_.size() && gsel_[index]; ++index)
    ;
  if (index < gsel_.size()) {
    gsel_[index] = true;
    g->SetSind(index);
    g->SetStatus(GroupStatus::INSTATE);
    int offset = kGroupBegin + 6 * index;

    // with gsb=(Rsb, Tsb) as the augmented state
    // augmentation is much simpler
    err_.segment<3>(offset) = err_.segment<3>(Index::Wsb);
    err_.segment<3>(offset + 3) = err_.segment<3>(Index::Tsb);

    P_.block(offset, 0, 3, err_.size()) =
        P_.block(Index::Wsb, 0, 3, err_.size());
    P_.block(0, offset, err_.size(), 3) =
        P_.block(0, Index::Wsb, err_.size(), 3);

    P_.block(offset + 3, 0, 3, err_.size()) =
        P_.block(Index::Tsb, 0, 3, err_.size());
    P_.block(0, offset + 3, err_.size(), 3) =
        P_.block(0, Index::Tsb, err_.size(), 3);

    VLOG(0) << StrFormat("group #%d inserted @ %d/%d", g->id(), index,
                               kMaxGroup);
  } else {
    throw std::runtime_error("Failed to find slot in state for group.");
  }
}

void Estimator::AddFeatureToState(FeaturePtr f) {
#ifndef NDEBUG
  CHECK(!f->instate()) << "feature already in state";
  CHECK(f->sind() == -1) << "feature slot already allocated";
#endif

  // change the covariance and error state
  int index;
  // find empty slot
  for (index = 0; index < fsel_.size() && fsel_[index]; ++index)
    ;
  if (index < fsel_.size()) {
    fsel_[index] = true;
    f->SetStatus(FeatureStatus::INSTATE);
    f->SetSind(index);
    f->FillCovarianceBlock(P_);
    VLOG(0) << StrFormat("feature #%d inserted @ %d/%d", f->id(), index,
                               kMaxFeature);
  } else {
    throw std::runtime_error("Failed to find slot in state for feature.");
  }
}

void Estimator::PrintErrorStateNorm() {
  VLOG(0) << StrFormat(
      "|Wsb|=%0.8f, |Tsb|=%0.8f, |Vsb|=%0.8f, "
      "|bg|=%0.8f, |ba|=%0.8f, |Wbc|=%0.8f, |Tbc|=%0.8f, |Wsg|=%0.8f\n",
      err_.segment<3>(Index::Wsb).norm(), err_.segment<3>(Index::Tsb).norm(),
      err_.segment<3>(Index::Vsb).norm(), err_.segment<3>(Index::bg).norm(),
      err_.segment<3>(Index::ba).norm(), err_.segment<3>(Index::Wbc).norm(),
      err_.segment<3>(Index::Tbc).norm(), err_.segment<2>(Index::Wsg).norm());
  for (auto g : instate_groups_) {
#ifndef NDEBUG
    CHECK(gsel_[g->sind()]) << "instate group not actually instate";
#endif
    VLOG(0) << StrFormat(
        "g#%d |W|=%0.8f, |T|=%0.8f\n", g->id(),
        err_.segment<3>(kGroupBegin + 6 * g->sind()).norm(),
        err_.segment<3>(kGroupBegin + 6 * g->sind() + 3).norm());
  }
  for (auto f : instate_features_) {
#ifndef NDEBUG
    CHECK(fsel_[f->sind()]) << "instate feature not yet instate";
#endif
    VLOG(0) << StrFormat(
        "f#%d |X|=%0.8f\n", f->id(),
        err_.segment<3>(kFeatureBegin + 3 * f->sind()).norm());
  }
}

void Estimator::AbsorbError(const VecX &err) {
  // motion state
  this->UpdateState(err.head<kMotionSize>());

#ifdef USE_ONLINE_IMU_CALIB
  // update IMU state
  Eigen::Matrix<number_t, 15, 1> dCaCg;
  dCaCg << err.segment<6>(Index::Ca), err.segment<9>(Index::Cg);
  imu_.UpdateState(dCaCg);
#endif

#ifdef USE_ONLINE_CAMERA_CALIB
  // update camera instrinsics
  Camera::instance()->UpdateState(
      err.segment<kMaxCameraIntrinsics>(kCameraBegin));
#endif
  // Camera::instance()->Print(std::cout);
  // std::cout << "Ca=\n" << imu_.Ca() << std::endl;
  // std::cout << "Cg=\n" << imu_.Cg() << std::endl;
  // std::cout << "td=" << err(Index::td) << std::endl;

  // augmented state
  for (auto g : instate_groups_) {
#ifndef NDEBUG
    CHECK(g->sind() != -1);
#endif
    int offset = kGroupBegin + 6 * g->sind();
    g->UpdateState(err.segment<6>(offset));

    // if (g->id() == gauge_group_) {
    //   std::cout << "gauge group:" << err.segment<6>(offset).transpose() <<
    //   std::endl;
    // }
  }
  for (auto f : instate_features_) {
#ifndef NDEBUG
    CHECK(f->sind() != -1);
#endif
    int offset = kFeatureBegin + 3 * f->sind();
    f->UpdateState(err.segment<3>(offset));
  }
}

void Estimator::AbsorbError() {
  AbsorbError(err_);
  err_.setZero();
}

void Estimator::MaintainBuffer() {
  if (!buf_.initialized) {
    if (buf_.size() >= InternalBuffer::MAX_SIZE) {
      std::make_heap(buf_.begin(), buf_.end(), cmp);
      buf_.initialized = true;
    }
  } else {
    std::push_heap(buf_.begin(), buf_.end(), cmp);
  }

  if (!async_run_) {
    // execute here
    if (buf_.initialized && buf_.size() > InternalBuffer::MAX_SIZE) {
      buf_.front()->Execute(this);
      std::pop_heap(buf_.begin(), buf_.end(), cmp);
      buf_.pop_back();
    }
  }
}

void Estimator::VisualMeas(const timestamp_t &ts_raw, const cv::Mat &img) {
  timestamp_t ts{ts_raw};
#ifdef USE_ONLINE_TEMPORAL_CALIB
  if (X_.td >= 0) {
    ts += timestamp_t(uint64_t(X_.td * 1e9)); // seconds -> nanoseconds
  } else {
    ts -= timestamp_t(uint64_t(-X_.td * 1e9)); // seconds -> nanoseconds
  }
#endif
  if (async_run_) {
    std::scoped_lock lck(buf_.mtx);
    buf_.push_back(std::make_unique<internal::Visual>(ts, img));
    MaintainBuffer();
  } else {
    buf_.push_back(std::make_unique<internal::Visual>(ts, img));
    MaintainBuffer();
  }
}

void Estimator::VisualMeasTrackerOnly(const timestamp_t &ts_raw, const cv::Mat &img) {
  timestamp_t ts{ts_raw};
#ifdef USE_ONLINE_TEMPORAL_CALIB
  if (X_.td >= 0) {
    ts += timestamp_t(uint64_t(X_.td * 1e9)); // seconds -> nanoseconds
  } else {
    ts -= timestamp_t(uint64_t(-X_.td * 1e9)); // seconds -> nanoseconds
  }
#endif
  if (async_run_) {
    std::scoped_lock lck(buf_.mtx);
    buf_.push_back(std::make_unique<internal::VisualTrackerOnly>(ts, img));
    MaintainBuffer();
  } else {
    buf_.push_back(std::make_unique<internal::VisualTrackerOnly>(ts, img));
    MaintainBuffer();
  }
}


void Estimator::VisualMeasPointCloud(
  const timestamp_t &ts_raw,
  const VecXi &feature_ids,
  const MatX2 &xp_vals)
{
  timestamp_t ts{ts_raw};
#ifdef USE_ONLINE_TEMPORAL_CALIB
  if (X_.td >= 0) {
    ts += timestamp_t(uint64_t(X_.td * 1e9)); // seconds -> nanoseconds
  } else {
    ts -= timestamp_t(uint64_t(-X_.td * 1e9)); // seconds -> nanoseconds
  }
#endif
  if (async_run_) {
    std::scoped_lock lck(buf_.mtx);
    buf_.push_back(std::make_unique<internal::VisualPointCloud>(
      ts, feature_ids, xp_vals));
    MaintainBuffer();
  } else {
    buf_.push_back(std::make_unique<internal::VisualPointCloud>(
      ts, feature_ids, xp_vals));
    MaintainBuffer();
  }
}


void Estimator::VisualMeasPointCloudTrackerOnly(
  const timestamp_t &ts_raw, 
  const VecXi &feature_ids,
  const MatX2 &xp_vals)
{
  timestamp_t ts{ts_raw};
#ifdef USE_ONLINE_TEMPORAL_CALIB
  if (X_.td >= 0) {
    ts += timestamp_t(uint64_t(X_.td * 1e9)); // seconds -> nanoseconds
  } else {
    ts -= timestamp_t(uint64_t(-X_.td * 1e9)); // seconds -> nanoseconds
  }
#endif
  if (async_run_) {
    std::scoped_lock lck(buf_.mtx);
    buf_.push_back(std::make_unique<internal::VisualPointCloudTrackerOnly>(
      ts, feature_ids, xp_vals));
    MaintainBuffer();
  } else {
    buf_.push_back(std::make_unique<internal::VisualPointCloudTrackerOnly>(
      ts, feature_ids, xp_vals));
    MaintainBuffer();
  }

}



void Estimator::InertialMeas(const timestamp_t &ts, const Vec3 &gyro,
                             const Vec3 &accel) {
  if (async_run_) {
    std::scoped_lock lck(buf_.mtx);
    buf_.push_back(std::make_unique<internal::Inertial>(ts, gyro, accel));
    MaintainBuffer();
  } else {
    buf_.push_back(std::make_unique<internal::Inertial>(ts, gyro, accel));
    MaintainBuffer();
  }
}


void Estimator::VisualMeasInternalTrackerOnly(const timestamp_t &ts, const cv::Mat &img) {
  if (!GoodTimestamp(ts))
    return;

  if (simulation_) {
    throw std::invalid_argument(
        "function VisualMeas cannot be called in simulation");
  }

  ++vision_counter_;
  timer_.Tick("visual-meas-tracker-only");
  UpdateSystemClock(ts);

  if (use_canvas_) {
    Canvas::instance()->Update(img);
  }
  auto tracker = Tracker::instance();

  // track features
  timer_.Tick("track");
  tracker->Update(img);
  timer_.Tock("track");
  // process features
  timer_.Tick("process-tracks");

  if (use_canvas_) {
    for (auto f : tracker->features_)
      Canvas::instance()->Draw(f);
  }

  for (auto it = tracker->features_.begin(); it != tracker->features_.end();) {
    auto f = *it;
    if (f->track_status() == TrackStatus::REJECTED ||
        f->track_status() == TrackStatus::DROPPED)
    {
      it = tracker->features_.erase(it);
      Feature::Destroy(f);
    } else {
      ++it;
    }
  }

  static int print_counter{0};
  if (print_timing_ && ++print_counter % 50 == 0) {
    std::cout << print_counter << std::endl;
    std::cout << timer_;
  }

  // Save the frame (only if set to true in json file)
  Canvas::instance()->SaveFrame();

  timer_.Tock("process-tracks");

  if (gauge_group_ == -1) {
    SwitchRefGroup();
  }
  timer_.Tock("visual-meas-tracker-only");
}

void Estimator::VisualMeasInternal(const timestamp_t &ts, const cv::Mat &img) {
  if (!GoodTimestamp(ts)) {
    std::cout << "Dropping a visual frame because its timestamp was delayed too far back in the past. Make MESSAGE_BUFFER_SIZE bigger." << std::endl;
    return;
  }
  if (simulation_) {
    throw std::invalid_argument(
        "function VisualMeas cannot be called in simulation");
  }

  ++vision_counter_;
  timer_.Tick("visual-meas");
  UpdateSystemClock(ts);
  if (vision_initialized_) {
    // propagate state upto current timestamp
    Propagate(true);
    if (use_canvas_) {
      Canvas::instance()->Update(img);
    }
    // measurement prediction for feature tracking
    auto tracker = Tracker::instance();
    Predict(tracker->features_);
    // track features
    timer_.Tick("track");
    tracker->Update(img);
    timer_.Tock("track");
    // process features
    timer_.Tick("process-tracks");
    ProcessTracks(ts, tracker->features_);
    timer_.Tock("process-tracks");

    if (gauge_group_ == -1) {
      SwitchRefGroup();
    }
  }
  timer_.Tock("visual-meas");
}


void Estimator::VisualMeasPointCloudInternal(
  const timestamp_t &ts,
  const VecXi &feature_ids,
  const MatX2 &xps)
{
  if (!GoodTimestamp(ts))
    return;

  if (!simulation_) {
    throw std::invalid_argument(
        "function VisualMeasPointCloud is only for simulation");
  }

  ++vision_counter_;
  timer_.Tick("visual-meas");
  UpdateSystemClock(ts);
  if (vision_initialized_) {
    // propagate state upto current timestamp
    Propagate(true);
    if (use_canvas_) {
      Canvas::instance()->UpdatePointCloud(xps);
    }
    // measurement prediction for feature tracking
    auto tracker = Tracker::instance();
    Predict(tracker->features_);
    // track features
    timer_.Tick("track");
    tracker->UpdatePointCloud(feature_ids, xps);
    timer_.Tock("track");
    // process features
    timer_.Tick("process-tracks");
    ProcessTracks(ts, tracker->features_);
    timer_.Tock("process-tracks");

    if (gauge_group_ == -1) {
      SwitchRefGroup();
    }
  }
  timer_.Tock("visual-meas");
}


void Estimator::VisualMeasPointCloudInternalTrackerOnly(
  const timestamp_t &ts,
  const VecXi &feature_ids,
  const MatX2 &xps)
{
  if (!GoodTimestamp(ts))
    return;

  if (!simulation_) {
    throw std::invalid_argument(
        "function VisualMeasPointCloud is only for simulation");
  }

  ++vision_counter_;
  timer_.Tick("visual-meas-tracker-only");
  UpdateSystemClock(ts);

  if (use_canvas_) {
    Canvas::instance()->UpdatePointCloud(xps);
  }

  auto tracker = Tracker::instance();
  // track features
  timer_.Tick("track");
  tracker->UpdatePointCloud(feature_ids, xps);
  timer_.Tock("track");

  if (use_canvas_) {
    for (auto f: tracker->features_)
      Canvas::instance()->Draw(f);
  }

  for (auto it = tracker->features_.begin(); it != tracker->features_.end();) {
    auto f = *it;
    if (f->track_status() == TrackStatus::REJECTED ||
        f->track_status() == TrackStatus::DROPPED)
    {
      it = tracker->features_.erase(it);
      Feature::Destroy(f);
    } else {
      ++it;
    }
  }

  // Save the frame (only if set to true in json file)
  Canvas::instance()->SaveFrame();

  if (gauge_group_ == -1) {
    SwitchRefGroup();
  }

  timer_.Tock("visual-meas-tracker-only");
}


void Estimator::Predict(std::list<FeaturePtr> &features) {
  for (auto f : features) {
    f->Predict(gsb(), gbc());
  }
}

void Estimator::UpdateJosephForm() {

  S_ = H_ * P_ * H_.transpose();

  for (int i = 0; i < diagR_.size(); ++i) {
    S_(i, i) += diagR_(i);
  }

  K_.setZero(err_.size(), H_.rows());
  K_.transpose() = S_.ldlt().solve(H_ * P_);
  err_ = K_ * inn_;

  // I_KH_.noalias() = -K_ * H_;
  // for (int i = 0; i < err_.size(); ++i) {
  //   I_KH_(i, i) += 1;
  // }

  // Here, I_KH is actually KH - I, but since
  // update of P is quadratic in I_KH, so it does not matter.
  I_KH_ = K_ * H_;
  for (int i = 0; i < err_.size(); ++i) {
    I_KH_(i, i) -= 1;
  }
  P_ = I_KH_ * P_ * I_KH_.transpose();

  int kr = K_.rows();
  int kc = K_.cols();
  for (int i = 0; i < kc; ++i) {
    K_.block(0, i, kr, 1) *= sqrt(diagR_(i));
  }
  P_.noalias() +=  K_ * K_.transpose();
}

std::tuple<number_t, bool> Estimator::HuberOnInnovation(const Vec2 &inn,
                                                     number_t Rviz) {

  number_t robust_Rviz{Rviz}; // robustified measurement variance
  bool outlier{false};     // consider this measurement as an outlier?

  if (number_t ratio{inn.squaredNorm() / (2 * Rviz) / outlier_thresh_};
      ratio > 1.0) {
    ratio = sqrt(ratio);
    robust_Rviz *= ratio;
    outlier = true;
    // outlier_counter += ratio;
  } else {
    // outlier_counter = 0
  }
  return std::make_tuple(robust_Rviz, outlier);
}

std::vector<FeaturePtr>
Estimator::FindNewOwnersForFeaturesOf(const std::vector<GroupPtr> &discards) {
  std::vector<FeaturePtr> nullref_features;
  Graph& graph{*Graph::instance()};
  for (auto g : discards) {
    // transfer ownership of the remaining features whose reference is this one
    auto failed = graph.TransferFeatureOwnership(
      g, gbc(), feature_owner_change_cov_factor_);
    nullref_features.insert(nullref_features.end(), failed.begin(),
                            failed.end());

  }
  MakePtrVectorUnique(nullref_features);
  return nullref_features;
}


void Estimator::DiscardGroups(const std::vector<GroupPtr> &discards) {
  Graph& graph{*Graph::instance()};
  for (auto g: discards) {
    if (g->id() == gauge_group_) {
      // just lost the gauge group
      gauge_group_ = -1;
    }
#ifdef USE_MAPPER
    Mapper::instance()->AddGroup(g, graph.GetGroupAdj(g));
#endif
    graph.RemoveGroup(g);
    if (g->instate()) {
      RemoveGroupFromState(g);
    }
    Group::Deactivate(g);
  }
}


void Estimator::DiscardFeatures(const std::vector<FeaturePtr> &discards) {
  Graph &graph{*Graph::instance()};
  for (auto f : discards) {
#ifdef USE_MAPPER
    Mapper::instance()->AddFeature(f, graph.GetFeatureAdj(f), gbc());
#endif
    graph.RemoveFeature(f);
    if (f->instate()) {
      RemoveFeatureFromState(f);
    }
    Feature::Deactivate(f);
  }
}


void Estimator::DestroyFeatures(const std::vector<FeaturePtr> &destroys) {
  Graph::instance()->RemoveFeatures(destroys);
  for (auto f : destroys) {
    if (f->instate()) {
      RemoveFeatureFromState(f);
    }
    Feature::Destroy(f);
  }
}

void Estimator::SwitchRefGroup() {
  auto candidates = Graph::instance()->GetInstateGroups();
  if (!candidates.empty()) {
    // FIXME: in addition to the variance, also take account of the number of
    // instate features
    // associated with the group -- for an efficient implementation, use a
    // decorator to get the
    // "number of instate features" attribute first
    GroupPtr g = FindNewRefGroup(candidates);

    // reset new gauge group
    //GroupPtr g{*git};
    gauge_group_ptr_ = g;
    gauge_group_ = g->id();
    g->SetStatus(GroupStatus::GAUGE);
    VLOG(0) << "gauge group #" << gauge_group_ << " selected";
    // std::cout << "gauge group #" << gauge_group_ << " selected";

    // now fix covariance of the new gauge group. This prevents the group's
    // state from changing.
    int offset = kGroupBegin + 6 * g->sind();
    if (group_degrees_fixed_ == 4) {
      P_.block(offset+2, 0, 4, err_.size()).setZero();
      P_.block(0, offset+2, err_.size(), 4).setZero();
    } else {
      P_.block(offset, 0, 6, err_.size()).setZero();
      P_.block(0, offset, err_.size(), 6).setZero();
    }
  }
}


GroupPtr Estimator::FindNewRefGroup(std::vector<GroupPtr>&candidates) {
  auto git = std::min_element(candidates.begin(), candidates.end(),
                         [this](const GroupPtr g1, const GroupPtr g2) -> bool {
                           int offset1 = kGroupBegin + 6 * g1->sind();
                           int offset2 = kGroupBegin + 6 * g2->sind();
                           number_t cov1{0}, cov2{0};
                           for (int i = 0; i < 6; ++i) {
                             cov1 += P_(offset1 + i, offset1 + i);
                             cov2 += P_(offset2 + i, offset2 + i);
                           }
                           return cov1 < cov2;
                         });
  return *git;
}


void Estimator::BackupState(std::unordered_set<FeaturePtr>& features,
                            std::unordered_set<GroupPtr>& groups)
{
  X0_ = X_;
  P0_ = P_;
  for (auto g : groups) {
    g->BackupState();
  }
  for (auto f : features) {
    f->BackupState();
  }
#ifdef USE_ONLINE_IMU_CALIB
  imu_.BackupState();
#endif

#ifdef USE_ONLINE_CAMERA_CALIB
  Camera::instance()->BackupState();
#endif
}


void Estimator::RestoreState(std::unordered_set<FeaturePtr>& features,
                            std::unordered_set<GroupPtr>& groups)
{
  X_ = X0_;
  P_ = P0_;
  for (auto f : features) {
    f->RestoreState();
  }
  for (auto g : groups) {
    g->RestoreState();
  }
#ifdef USE_ONLINE_IMU_CALIB
  imu_.RestoreState();
#endif

#ifdef USE_ONLINE_CAMERA_CALIB
  Camera::instance()->RestoreState();
#endif
}

bool Estimator::FeatureCovComparison(FeaturePtr f1, FeaturePtr f2) const {
  number_t score1 = InstateFeatureCov(f1).norm();
  number_t score2 = InstateFeatureCov(f2).norm();
  return (score1 <= score2);
}


bool Estimator::FeatureCovXYComparison(FeaturePtr f1, FeaturePtr f2) const {
  number_t score1 = InstateFeatureCov(f1).block<2,2>(0,0).norm();
  number_t score2 = InstateFeatureCov(f2).block<2,2>(0,0).norm();
  return (score1 <= score2);
}


bool Estimator::UsingLoopClosure() const {
#ifdef USE_MAPPER
  return Mapper::instance()->UseLoopClosure();
#else
  return false;
#endif
}


void Estimator::FixFeatureXY(FeaturePtr f) {
  int foff = kFeatureBegin + 3*f->sind();
  P_.block(foff, 0, 2, err_.size()).setZero();
  P_.block(0, foff, err_.size(), 2).setZero();
}


} // xivo
