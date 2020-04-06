// Inertial-aided Visual Odometry estimator.
// Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <ostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Eigen/Sparse"
#include "opencv2/core/core.hpp"
#include "json/json.h"

#include "component.h"
#include "core.h"
#include "graph.h"
#include "imu.h"
#include "tracker.h"
#include "visualize.h"

namespace xivo {

class Estimator;
using EstimatorPtr = Estimator*;
EstimatorPtr CreateSystem(const Json::Value &cfg);


namespace internal {
class Message {
public:
  Message(const timestamp_t &ts) : ts_{ts} {}
  const timestamp_t &ts() const { return ts_; }
  virtual ~Message() = default;
  virtual void Execute(EstimatorPtr) {}

protected:
  timestamp_t ts_;
};

class Visual : public Message {
public:
  Visual(const timestamp_t &ts, const cv::Mat &img) : Message{ts}, img_{img} {}
  void Execute(EstimatorPtr est);

private:
  cv::Mat img_;
};

class Inertial : public Message {
public:
  Inertial(const timestamp_t &ts, const Vec3 &gyro, const Vec3 &accel)
      : Message{ts}, gyro_{gyro}, accel_{accel} {}
  void Execute(EstimatorPtr est);

private:
  Vec3 gyro_, accel_;
};

} // namespace internal


class Estimator : public Component<Estimator, State> {
  friend class internal::Visual;
  friend class internal::Inertial;

public:
  static EstimatorPtr Create(const Json::Value &cfg);
  static EstimatorPtr instance();

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ~Estimator();

  void Run();
  // process inertial measurements
  void InertialMeas(const timestamp_t &ts, const Vec3 &gyro, const Vec3 &accel);
  // perform tracking/matching to generate tracks
  void VisualMeas(const timestamp_t &ts, const cv::Mat &img);

  // accessors
  SE3 gbc() const { return SE3{X_.Rbc, X_.Tbc}; }
  SE3 gsb() const { return SE3{X_.Rsb, X_.Tsb}; }
  SE3 gsc() const { return gsb() * gbc(); }
  const State& X() const { return X_; }
  const timestamp_t &ts() const { return curr_time_; }
  MatX P() const { return P_; }
  MatX Pstate() const { return P_.block<kMotionSize,kMotionSize>(0,0); }
  MatX CameraCov() const {
#ifdef USE_ONLINE_CAMERA_CALIB
    return P_.block<kMaxCameraIntrinsics,kMaxCameraIntrinsics>(kCameraBegin,
      kCameraBegin);
#else
    Eigen::Matrix<number_t, 9, 9> all_zeros;
    return all_zeros;
#endif
  }
  Vec3 Vsb() const { return X_.Vsb; }
  Vec3 bg() const { return X_.bg; }
  Vec3 ba() const { return X_.ba; }
  SO3 Rg() const { return X_.Rg; }
  number_t td() const { return X_.td; }
  Mat3 Ca() const { return imu_.Ca(); }
  Mat3 Cg() const { return imu_.Cg(); }
  Vec3 inn_Wsb() const { return inn_.segment(Index::Wsb,3); }
  Vec3 inn_Tsb() const { return inn_.segment(Index::Tsb,3); }
  Vec3 inn_Vsb() const { return inn_.segment(Index::Vsb,3); }
  bool MeasurementUpdateInitialized() const { return MeasurementUpdateInitialized_; }
  int gauge_group() const { return gauge_group_; }
  int num_instate_features() const { return instate_features_.size(); };
  int num_instate_groups() const {return instate_groups_.size(); };
  MatX3 InstateFeaturePositions(int n_output) const;
  MatX3 InstateFeaturePositions() const;
  MatX6 InstateFeatureCovs(int n_output) const;
  MatX6 InstateFeatureCovs() const;
  MatX3 InstateFeatureXc(int n_output) const;
  MatX3 InstateFeatureXc() const;
  void InstateFeaturePositionsAndCovs(int max_output, int &npts,
    MatX3 &positions, MatX6 &covs, MatX2 &pixels, VecXi &feature_ids);
  VecXi InstateFeatureIDs(int n_output) const;
  VecXi InstateFeatureIDs() const;
  VecXi InstateFeatureSinds(int n_output) const;
  VecXi InstateFeatureSinds() const;
  VecXi InstateGroupIDs() const;
  MatX7 InstateGroupPoses() const;
  MatX InstateGroupCovs() const;
  VecXi InstateGroupSinds() const;

  int OOS_update_min_observations() { return OOS_update_min_observations_; }

private:
  void UpdateState(const State::Tangent &dX) { X_ += dX; }

  /** Top-level function for state prediction and update when an IMU packet
   *  arrives */
  void InertialMeasInternal(const timestamp_t &ts, const Vec3 &gyro,
                            const Vec3 &accel);

  /** Top-level function for state prediction and update when an image
   *  packet arrives */
  void VisualMeasInternal(const timestamp_t &ts, const cv::Mat &img);

  // initialize gravity with initial stationary samples
  bool InitializeGravity();
  /** Integrates the State `X`. If parameter `visual_meas` is set to `false`, we
   *  update `slope_accel_` and `slope_gyro_`. If `visual_meas` is set to `true`, we
   *  use `slope_accel_` and `slope_gyro` to adjust the last IMU measurement. */
  void Propagate(bool visual_meas);
  /** kalman filter update step -- uses Joseph form */
  void UpdateJosephForm();
  /** Predicts measurement (pixels) of features in input. */
  void Predict(std::list<FeaturePtr> &features);
  /** compute the motion jacobian F and G (private members `F_` and `G_`) at the
   *  given state and measurement. */
  void ComputeMotionJacobianAt(const State &X,
                               const Eigen::Matrix<number_t, 6, 1> &gyro_accel);
  // only need velocity as the slope for integration
  void ComposeMotion(State &X, const Vec3 &V,
                     const Eigen::Matrix<number_t, 6, 1> &gyro_accel, number_t dt);
  /** perform Fehlberg numerical integration */
  void Fehlberg(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  /** perform Prince-Dormand numerical integration */
  void PrinceDormand(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  /** Perform one-step in Prince-Dormand numerical integration and
   *  return max(slope), i.e., max(V, max(gyro), max(accel)) */
  number_t PrinceDormandStep(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  /** perform vanilla RK4 without step control */
  void RK4(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  /** perform one-step in RK4 integration (4 inner steps) */
  void RK4Step(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);

  void ProcessTracks(const timestamp_t &ts, std::list<FeaturePtr> &features);

  /** Function that contains logic for outlier rejection, filter EKF update, and
   *  filter MSCKF update. It will mark features for removal from the state, but
   *  does not do the actual removing and does not update the graph. */
  void Update();

  /** Outlier rejection on `Tracker` matches. Always occurs after MH-gating. */
  std::vector<FeaturePtr>
  OnePointRANSAC(const std::vector<FeaturePtr> &ic_matches);
  std::tuple<number_t, bool> HuberOnInnovation(const Vec2 &inn, number_t Rviz);

  void UpdateSystemClock(const timestamp_t &now);

  /** Checks that timestamp `now` (= timestamp of message currently processed) is
   *  at or later than timestamp of last processed message. */
  bool GoodTimestamp(const timestamp_t &now);

  // same as above, but the group list will be untouched
  void RemoveGroupFromState(GroupPtr g);
  void AddGroupToState(GroupPtr g);
  std::vector<FeaturePtr> DiscardGroups(const std::vector<GroupPtr> &discards);
  void DiscardFeatures(const std::vector<FeaturePtr> &discards);
  void SwitchRefGroup();

  // same as above, but the feature list will be untouched
  void RemoveFeatureFromState(FeaturePtr f);
  void AddFeatureToState(FeaturePtr f);

  void AbsorbError(const VecX &err); // absorb error state into nominal state
  void AbsorbError();                // absorb error state into nominal state
  // helpers
  void PrintErrorStateNorm();
  void PrintErrorState();
  void PrintNominalState();


private:
  Estimator(const Json::Value &cfg);
  static std::unique_ptr<Estimator> instance_;

private:
  std::vector<FeaturePtr> instate_features_; ///< in-state features
  std::vector<FeaturePtr> oos_features_;     ///< out-of-state features
  std::vector<GroupPtr> instate_groups_;     ///< in-state groups

  /** Index of the current gauge group. It is set to -1 when we lose the current
   *  gauge group while calling `ProcessTracks`. */
  int gauge_group_;

private:
  Config cfg_;        // this is just a reference of the global parameter server
  bool simulation_;   // estimator used in simulation or not
  bool use_canvas_;   // visualization or not
  bool print_timing_; // show timing info
  std::string integration_method_; ///< motion integration numerical scheme

  /** Whether or not to sue 1-pt RANSAC in outlier rejection. */
  bool use_1pt_RANSAC_;
  number_t ransac_thresh_, ransac_prob_, ransac_Chi2_;

  /** Whether or not to use MSCKF measurement update */
  bool use_OOS_;
  bool use_compression_;            // measurement compression
  number_t compression_trigger_ratio_; // use measurement compression, if the ratio
                                    // of columns/rows of measurement matrix is
                                    // above this level
  /** Minimum number of observations a feature needs by the time `Tracker` drops
   *  it in order to use it in a MSCKF update. */
  int OOS_update_min_observations_;
  bool use_depth_opt_;              // use depth optimization or not
  RefinementOptions refinement_options_; // depth refinement options
  SubfilterOptions subfilter_options_;   // depth-subfilter options
  bool triangulate_pre_subfilter_; // depth triangulation before depth subfilter
  TriangulateOptions triangulate_options_;

  /** Minimum number of steps a feature is an outlier before it is removed */
  int remove_outlier_counter_;

  /** The current state estimate. Contains nominal state and calibrations, but no
   *  feature positions. */
  State X_;
  /** Filter's error state: Contains both pose and feature positions. */
  VecX err_;
  /** Whether or not each group is in-state */
  std::array<bool, kMaxGroup> gsel_;
  /** Whether or not each feature is in-state */
  std::array<bool, kMaxFeature> fsel_;
  /** Data and operators for IMU calibration variables `Ca` and `Cg` */
  IMU imu_;
  /** Current estimate of the gravity vector resolved in the reference frame. */
  Vec3 g_;

  // measurement noise

  /** The initial depth value given to new features when they are first created. It is
   *  updated at every frame to be (almost) equal to the median depth of all the
   *  features currently in the state. (i.e. `init_z = 0.01*init_z + 0.99*median_depth`) */
  number_t init_z_;
  /** Default subfilter covariance for each feature for x-coordinate at initialization.
   *  (Subfilter covariance is initialized as a diagonal.) */
  number_t init_std_x_;
  /** Default subfilter covariance for each feature for y-coordinate at initialization.
   *  (Subfilter covariance is initialized as a diagonal.) */
  number_t init_std_y_;
  /** Default subfilter covariance for each feature for z-coordinate at initialization.
   *  (Subfilter covariance is initialized as a diagonal.) */
  number_t init_std_z_;

  /** The minimum depth that a feature can be given when it is first
   *  created. (i.e. minimum value of `init_z_`) */
  number_t min_z_;
  /** The maximum depth that a feature can be given when it is first
   *  created. (i.e. maximum value of `init_z_`) */
  number_t max_z_;

  /** Error state dynamics Jacobian; Used for covariance update in EKF's
   *  prediction step. */
  Eigen::SparseMatrix<number_t> F_;
  /** Error state noise input-matrix Jacobian; Used for covariance update in EKF's
   *  prediction step. */
  Eigen::SparseMatrix<number_t> G_;
  /** Filter covariance. Size grows and shrinks with the number of tracked
   *  features. */
  MatX P_;
  /** Filter motion covariance. Size is `kMotionSize` x `kMotionSize` */
  MatX Qmodel_;
  /** 
   * Filter IMU measurement covaraince, made up of four 3x3 blocks for a total
   * dimention of 12 x 12. The four blocks correspond to the gyro,
   * accelerometer, gyro bias, and accelerometer bias, measurements,
   * respectively. */
  MatX Qimu_;

  // for clamping signals
  bool clamp_signals_;
  Vec3 max_gyro_;
  Vec3 max_accel_;

  // for update

  /** Set to true once update has been initialized */
  bool MeasurementUpdateInitialized_;
  /** Filter predicted covariance */
  MatX S_;
  /** Filter Kalman gain */
  MatX K_;
  /** Filter measurement Jacobian */
  MatX H_;
  /** The matrix `I-K*H` used in the Kalman Filter measurement update */
  MatX I_KH_;
  /** Filter innovation */
  VecX inn_;
  /** Diagonal of visual feature measurement covariance used in the filter.
   *  (We assume that the feature measurement covariance is diagonal, as opposed
   *  to just positive semi-definite.) Its size is 2*(number of in-state inliers) +
   *  (MSCKF measurement size) */
  VecX diagR_;
  /** The filter's (assumed) measurement covariance of every element of each MSCKF
   *  measurement. */
  number_t Roos_;
  /** The filter's (assumed) measurement covariance of x and y pixel measurement for
   *  in-state tracked features. */
  number_t R_;
  number_t Rtri_;           // UNUSED? measurement covariance, depth sub-filter
  number_t outlier_thresh_; // outlier threshold -- multipler of the measurement
                         // variance

  // MH (Mahalanobis) gating parameters
  bool use_MH_gating_;
  number_t MH_thresh_;           // MH threshold
  int min_required_inliers_;  // minimal inliers needed to perform update
  number_t MH_thresh_multipler_; // if not enough inliers, repeatedly multiple the
                              // MH_thresh by this amount

  // time
  timestamp_t last_imu_time_, curr_imu_time_; // time when the imu meas arrives
  timestamp_t last_vision_time_,
      curr_vision_time_;  // time when visual meas arrives
  timestamp_t curr_time_; // current system time
  timestamp_t last_time_; // last measurement time, either imu or visual

  Vec3 curr_accel_, curr_gyro_; // current gyro and accel measurement
  Vec3 last_accel_, last_gyro_; // accel & gyro measurement at last_time
  Vec3 slope_accel_, slope_gyro_;

  bool gravity_initialized_, vision_initialized_;
  int imu_counter_, vision_counter_;

  // helpers
  int gravity_init_counter_;
  std::vector<Vec3> gravity_init_buf_; // buffer of accel measurements for
                                       // gravity initialization
  // measurements buffer
  struct InternalBuffer
      : public std::vector<std::unique_ptr<internal::Message>> {
    static constexpr int MAX_SIZE = 10;
    InternalBuffer() : initialized{false} {}
    std::mutex mtx;
    bool initialized;
  } buf_;
  bool async_run_; // if true, run in a separate thread
  void MaintainBuffer();

  own<std::thread *> worker_;

  /** Computes the running average time for dynamics propatagion, visual measurement
   *  processing, tracker, update tracker, jacobian, MH gating. (Those quantities
   *  overlap). */
  Timer timer_;
  std::unique_ptr<std::default_random_engine> rng_;
};

} // xivo
