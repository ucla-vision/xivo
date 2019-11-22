// The feature class.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <functional>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "component.h"
#include "core.h"
#include "jac.h"
#include "options.h"
#include "project.h"

namespace xivo {


/** Track is a C++ <vector> containing all the (x,y) pixel detections found by
 *  the `Tracker` over a set of consecutive images paired with some metadata.
 */
class Track : public std::vector<Vec2, Eigen::aligned_allocator<Vec2>> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Track() : status_(TrackStatus::CREATED) {}
  Track(number_t x, number_t y) { Reset(x, y); }

  /** Deletes the entire history of tracks and starts a new vector. */
  void Reset(number_t x, number_t y) {
    clear();
    status_ = TrackStatus::CREATED;
    push_back(Vec2(x, y));
  }

  TrackStatus status() const { return status_; }
  void SetStatus(TrackStatus status) { status_ = status; }
  void SetDescriptor(const cv::Mat &descriptor) { descriptor_ = descriptor; }
  void SetKeypoint(const cv::KeyPoint &keypoint) { keypoint_ = keypoint; }
  const cv::KeyPoint &keypoint() const { return keypoint_; }
  cv::KeyPoint &keypoint() { return keypoint_; }
  const cv::Mat &descriptor() const { return descriptor_; }
  cv::Mat &descriptor() { return descriptor_; }

protected:
  /** CREATED, TRACKED, REJECTED, or DROPPED */
  TrackStatus status_;

  /** OpenCV Keypoint from when this track was first detected in `Tracker::Detect()` */
  cv::KeyPoint keypoint_;

  /** Descriptor of the very last keypoint. */
  cv::Mat descriptor_;
};


/** All the data associated with a single tracked feature.
 *  Essentially, the `Track` class plus
 *  - estimate of current 3D position with respect to the global reference frame
 *  - subfiltering and triangulation functions to accurately estimate depth
 *  - Functions to compute Jacobians for the `Estimator` class's measurement update.
 */
class Feature : public Component<Feature, Vec3>, public Track {
  friend class MemoryManager;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static FeaturePtr Create(number_t x, number_t y);
  static void Delete(FeaturePtr f);

  /** Appends another point to vector of observations.
   *  Recall: (`Feature` << `Track` << `std::vector` */
  void UpdateTrack(number_t x, number_t y) { emplace_back(x, y); }
  /** Appends another point to vector of observations.
   *  Recall: (`Feature` << `Track` << `std::vector` */
  void UpdateTrack(const Vec2 &pt) { UpdateTrack(pt(0), pt(1)); }

  /** Returns whether or not the feature is currently in the filter's state */
  bool instate() const;
  // score of the potential goodness of being an instate feature
  // The higher, the better.
  number_t score() const;
  number_t outlier_counter() const { return outlier_counter_; }
  /**
   * Gets actual depth of feature from variable `x_` (calculation is different
   * depending on whether or not we're using an inverse-depth or log-depth
   * parameterization).
   * \todo Ensure depth is positive when using inverse-depth parameterization,
   *       which is guaranteed when using log-depth paramterization. */
  number_t z() const;
  const Vec3 &x() const { return x_; }
  const Mat3 &P() const { return P_; }
  Vec3 &x() { return x_; }
  Mat3 &P() { return P_; }
  void BackupState() { x0_ = x_; }
  void RestoreState() { x_ = x0_; }
  // get 3D coordinates in reference camera frame
  Vec3 Xc(Mat3 *dXc_dx = nullptr);
  // get 3D coordinates in spatial frame, cam2body alignment is required
  Vec3 Xs(const SE3 &gbc, Mat3 *dXs_dx = nullptr);
  const Vec3& Xs() const { return Xs_; }

  // return (2M-3) as the dimension of the measurement
  /** Computes the Jacobian for the in-state (EKF) measurement model. */
  void ComputeJacobian(const Mat3 &Rsb, const Vec3 &Tsb, const Mat3 &Rbc,
                       const Vec3 &Tbc, const Vec3 &gyro, const Mat3 &Cg,
                       const Vec3 &bg, const Vec3 &Vsb, number_t td,
                       const VecX &error_state);

  int oos_inn_size() const { return oos_jac_counter_; }

  /** Computes the Jacobian for the out-of-state (MSCKF) measurement model. */
  int ComputeOOSJacobian(const std::vector<Obs> &obs, const Mat3 &Rbc,
                         const Vec3 &Tbc, const VecX &error_state);
  /** Contains the equations used in `Feature::ComputeOOSJacobian` for each
   *  observation.
   *  \todo make the following private */
  void ComputeOOSJacobianInternal(const Obs &obs, const Mat3 &Rbc,
                                  const Vec3 &Tbc, const VecX &error_state);

  // fill-in the corresponding jacobian block
  // H: the big jacobian matrix of all measurements
  // offset: of the block in H
  void FillJacobianBlock(MatX &H, int offset);
  // fill-in the corresponding covariance block when inserting the feature into state
  // P: the covariance matrix of the estimator
  void FillCovarianceBlock(MatX &P);

  const Eigen::Matrix<number_t, 2, kFullSize> &J() const { return J_; }
  const Vec2 &inn() const { return inn_; }

  /** Gets the last measurement (from the `Tracker`) of this feature */
  const Vec2 &xp() const { return back(); }
  /** Returns the last-computed predicted measurement
   *  (does not compute a new prediction) */
  const Vec2 &pred() const { return pred_; }
  /** Computes a new predicted measurement (in pixels) given transformations
   *  `gsb` and `gbc` */
  const Vec2 &Predict(const SE3 &gsb, const SE3 &gbc) {
    Vec3 Xc = (gsb * gbc).inv() * this->Xs(gbc);
    pred_ = Camera::instance()->Project(project(Xc));
    return pred_;
  }
  /** Sets variable `pred_`, the last computed predicted measurement to (-1,-1),
   *  the default "invalid" value for a predicted measurement. */
  void ResetPred() { pred_ << -1, -1; }

  ////////////////////////////////////////
  // OOS Jacobians accessors
  ////////////////////////////////////////
  VecX ro() const { return oos_.inn.head(oos_jac_counter_); }
  MatX Ho() const { return oos_.Hx.topRows(oos_jac_counter_); }

  void Initialize(number_t z0, const Vec3 &std_xyz);

  FeatureStatus status() const { return status_; }
  void SetStatus(FeatureStatus status) { status_ = status; }

  void SetTrackStatus(TrackStatus status) { Track::SetStatus(status); }
  TrackStatus track_status() const { return Track::status(); }

  int id() const { return id_; }
  int sind() const { return sind_; }
  void SetSind(int ind) { sind_ = ind; }

  int lifetime() const { return lifetime_; }
  int IncrementLifetime() { lifetime_++; }
  int ResetLifetime() { lifetime_ = 0; }

  GroupPtr ref() const { return ref_; }
  void SetRef(GroupPtr ref);
  void ResetRef(GroupPtr nref);

  // subfilter used for depth initialization
  void SubfilterUpdate(const SE3 &gsb, const SE3 &gbc,
                       const SubfilterOptions &options);
  bool RefineDepth(const SE3 &gbc, const std::vector<Obs> &obs,
                   const RefinementOptions &options);
  // triangulate the 3D point from the reference and another view
  void Triangulate(const SE3 &gsb, const SE3 &gbc,
                   const TriangulateOptions &options);

  void SetState(const Vec3 &x) { x_ = x; }
  void UpdateState(const Vec3 &dx) { x_ += dx; }

  /** Initial value of static variable `Feature::counter_`/smallest possible number
   *  used for feature IDs. Its purpose is so that feature IDs and group IDs never
   *  overlap.
   *  \todo Completely separate feature and group IDs so that we don't have
   *        problems if this SLAM code is ever run for a long, long time.  */
  static constexpr int counter0 = 10000;  

private:
  Feature(const Feature &) = delete;
  /** default constructor used memory manager's pre-allocation */
  Feature() = default;
  /** Resets a `Feature` object. Calls `Track::Reset` */
  void Reset(number_t x, number_t y);

private:
  /** Total number of features ever created (never decremented) +
   *  (static constexpr) `counter0`. Used for getting ID values of newly created
   *  features. `counter_` starts at `counter0` so that feature and group IDs
   *  do not overlap. */
  static int counter_;
  /** Feature ID. IDs are in order of creation. */
  int id_;
  /** Index of feature in Estimator's array of instate features. Not set until
   *  `status_` is `FeatureStatus::READY`. */
  int sind_;

  /** CREATED, INITIALIZING, READY, INSTATE, REJECTED_BY_FILTER, REJECTED_BY_TRACKER,
   *  or DROPPED */
  FeatureStatus status_;
  /** Pointer to feature's reference group: pose/instance in time where the feature
   *  is first observed. */
  GroupPtr ref_;

  /** Total number of timesteps (image frames) since the feature was first detected. */
  int lifetime_;

  /** Projected state: Let (X, Y, Z) be the coordinates of the feature in 3D
   *  space with respect to the current camera frame. Then, this variable
   *  contains the vector (X/Z, Y/Z, log(Z)) or (X/Z, Y/Z, 1/Z) when compiling
   *  with `#USE_INVDEPTH` */
  Vec3 x_;

  /** "Backup" of `Feature::x_` used in `Estimator::OnePointRANSAC` */
  Vec3 x0_;

  /** Subfilter (for estimating depth) covariance */
  Mat3 P_;

  /** Predicted pixel coordinates - computed right before the `Estimator` class's
   *  measurement update step in `Feature::Predict`. */
  Vec2 pred_;

  /** 3D coordinates of the feature with respect to the current camera frame. */
  Vec3 Xc_;

  /** 3D coordiantes of the feature with respect to the (current) global reference
   *  frame. */
  Vec3 Xs_;

  Eigen::Matrix<number_t, 2, kFullSize> J_;

  /** `xp` - predicted observation used in the filter for this particular feature. */
  Vec2 inn_;

  /** Measurement model Jacobian with respect to the error state used in the filter. */
  Mat23 Hx_;

  // outlier rejection
  int init_counter_;
  bool inlier_;
  number_t outlier_counter_;

  /** Contains current intermediate variables used to compute the Jacobians in both the
   *  EKF and MSCKF measurement models. */
  static JacobianCache cache_;

  /** Current MSCKF measurement Jacobians (both Hf and Hx) and innovation */
  OOSJacobian oos_;

  /** Number of measurements in the MSCKF measurement update. */
  int oos_jac_counter_;

#ifdef APPROXIMATE_INIT_COVARIANCE
  // correlation block between local feature state (x) and group pose
  std::unordered_map<int, Eigen::Matrix<number_t, kFeatureSize, kGroupSize>> cov_;
  // correlation block between local feature state (x) and camera-body alignment (c),
  // and reference group (r)
  Eigen::Matrix<number_t, kFeatureSize, kGroupSize> cov_xc_, cov_xr_;
#endif


public:
  // simulation
  struct {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 Xs;
    Vec2 xp, xc;
    number_t z;
    int lifetime;
  } sim_;

};

} // namespace xivo
