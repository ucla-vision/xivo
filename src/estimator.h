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

namespace internal {
class Message {
public:
  Message(const timestamp_t &ts) : ts_{ts} {}
  const timestamp_t &ts() const { return ts_; }
  virtual ~Message() = default;
  virtual void Execute(Estimator *) {}

protected:
  timestamp_t ts_;
};

class Visual : public Message {
public:
  Visual(const timestamp_t &ts, const cv::Mat &img) : Message{ts}, img_{img} {}
  void Execute(Estimator *est);

private:
  cv::Mat img_;
};

class Inertial : public Message {
public:
  Inertial(const timestamp_t &ts, const Vec3 &gyro, const Vec3 &accel)
      : Message{ts}, gyro_{gyro}, accel_{accel} {}
  void Execute(Estimator *est);

private:
  Vec3 gyro_, accel_;
};

} // namespace internal

class Estimator : public Component<Estimator, State> {
  friend class internal::Visual;
  friend class internal::Inertial;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Estimator(const Json::Value &cfg);
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
  State X() const { return X_; }
  const timestamp_t &ts() const { return curr_time_; }
  bool Finished();

private:
  void UpdateState(const State::Tangent &dX) { X_ += dX; }

  void InertialMeasInternal(const timestamp_t &ts, const Vec3 &gyro,
                            const Vec3 &accel);

  void VisualMeasInternal(const timestamp_t &ts, const cv::Mat &img);

  // initialize gravity with initial stationary samples
  bool InitializeGravity();
  // propagate state
  void Propagate(bool visual_meas);
  // kalman filter update -- joseph form
  void UpdateJosephForm();
  // measurement prediction
  void Predict(std::list<FeaturePtr> &features);
  // compute the motion jacobian F and G at the given linearization point
  void ComputeMotionJacobianAt(const State &X,
                               const Eigen::Matrix<number_t, 6, 1> &gyro_accel);
  // only need velocity as the slope for integration
  void ComposeMotion(State &X, const Vec3 &V,
                     const Eigen::Matrix<number_t, 6, 1> &gyro_accel, number_t dt);
  // perform Fehlberg numerical integration
  void Fehlberg(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  // perform Prince-Dormand numerical integration
  void PrinceDormand(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  // return max(slope), i.e., max(V, max(gyro), max(accel))
  number_t PrinceDormandStep(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  // perform vanilla RK4 without step control
  void RK4(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  // perform one-step in RK4 integration (4 inner steps)
  void RK4Step(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);

  void ProcessTracks(const timestamp_t &ts, std::list<FeaturePtr> &features);
  void Update();
  // one point RANSAC on individually compatible matches
  std::vector<FeaturePtr>
  OnePointRANSAC(const std::vector<FeaturePtr> &ic_matches);
  std::tuple<number_t, bool> HuberOnInnovation(const Vec2 &inn, number_t Rviz);

  void UpdateSystemClock(const timestamp_t &now);
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
  std::vector<FeaturePtr> instate_features_; // in-state features
  std::vector<FeaturePtr> oos_features_;     // out-of-state features
  std::vector<GroupPtr> instate_groups_;     // in-state groups
  // Graph graph_;
  int gauge_group_; // index of the selected gauge group, -1 for none

private:
  Config cfg_;        // this is just a reference of the global parameter server
  bool simulation_;   // estimator used in simulation or not
  bool use_canvas_;   // visualization or not
  bool print_timing_; // show timing info
  std::string integration_method_; // motion integration numerical scheme
  bool use_1pt_RANSAC_;            // one point ransac
  number_t ransac_thresh_, ransac_prob_, ransac_Chi2_;
  bool use_OOS_;                    // Out-Of-State measurement update
  bool use_compression_;            // measurement compression
  number_t compression_trigger_ratio_; // use measurement compression, if the ratio
                                    // of columns/rows of measurement matrix is
                                    // above this level
  bool use_depth_opt_;              // use depth optimization or not
  RefinementOptions refinement_options_; // depth refinement options
  SubfilterOptions subfilter_options_;   // depth-subfilter options
  bool triangulate_pre_subfilter_; // depth triangulation before depth subfilter
  TriangulateOptions triangulate_options_;

  State X_;
  VecX err_;                           // error state
  std::array<bool, kMaxGroup> gsel_;   // group selector
  std::array<bool, kMaxFeature> fsel_; // feature selector
  // imu component
  IMU imu_;
  // gravity
  Vec3 g_;

  // measurement noise
  number_t init_z_, init_std_x_, init_std_y_, init_std_z_;
  number_t min_z_, max_z_;

  // jacobians
  Eigen::SparseMatrix<number_t> F_, G_;
  // covariances
  MatX P_;
  MatX Qmodel_; // kMotionSize x kMotionSize
  MatX Qimu_;   // 12 x 12

  // for update
  MatX S_;
  MatX K_;               // kalman gain
  MatX H_;               // measurement Jacobian
  MatX I_KH_;            // I-KH
  VecX inn_;             // innovation stack
  VecX diagR_;           // diagonal of measurement covariance
  number_t Roos_;           // oos measurement covariance in pixels
  number_t R_;              // covariance of visual measurement noise, in pixels
  number_t Rtri_;           // measurement covariance, depth sub-filter
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

  // helplers
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

  Timer timer_;
  std::unique_ptr<std::default_random_engine> rng_;
};

} // xivo
