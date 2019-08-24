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


class Track : public std::vector<Vec2, Eigen::aligned_allocator<Vec2>> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Track() : status_(TrackStatus::CREATED) {}
  Track(number_t x, number_t y) { Reset(x, y); }

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
  TrackStatus status_;

  // initial keypoint to store extra info, e.g., octave, orientation, etc.
  cv::KeyPoint keypoint_;
  // last descriptor
  cv::Mat descriptor_;
};

class Feature : public Component<Feature, Vec3>, public Track {
  friend class MemoryManager;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static FeaturePtr Create(number_t x, number_t y);
  static void Delete(FeaturePtr f);

  void UpdateTrack(number_t x, number_t y) { emplace_back(x, y); }
  void UpdateTrack(const Vec2 &pt) { UpdateTrack(pt(0), pt(1)); }

  // quick predictor of instate status
  bool instate() const;
  // score of the potential goodness of being an instate feature
  // The higher, the better.
  number_t score() const;
  number_t outlier_counter() const { return outlier_counter_; }
  // get depth
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
  void ComputeJacobian(const Mat3 &Rsb, const Vec3 &Tsb, const Mat3 &Rbc,
                       const Vec3 &Tbc, const Vec3 &gyro, const Mat3 &Cg,
                       const Vec3 &bg, const Vec3 &Vsb, number_t td);

  int oos_inn_size() const { return oos_jac_counter_; }

  int ComputeOOSJacobian(const std::vector<Obs> &obs, const Mat3 &Rbc,
                         const Vec3 &Tbc);
  // FIXME: make the following private
  void ComputeOOSJacobianInternal(const Obs &obs, const Mat3 &Rbc,
                                  const Vec3 &Tbc);

  // fill-in the corresponding jacobian block
  // H: the big jacobian matrix of all measurements
  // offset: of the block in H
  void FillJacobianBlock(MatX &H, int offset);

  const Eigen::Matrix<number_t, 2, kFullSize> &J() const { return J_; }
  const Vec2 &inn() const { return inn_; }

  const Vec2 &xp() const { return back(); }
  const Vec2 &pred() const { return pred_; }
  const Vec2 &Predict(const SE3 &gsb, const SE3 &gbc) {
    Vec3 Xc = (gsb * gbc).inv() * this->Xs(gbc);
    pred_ = Camera::instance()->Project(project(Xc));
    return pred_;
  }
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

private:
  Feature(const Feature &) = delete;
  // default constructor used memory manager's pre-allocation
  Feature() = default;
  // Feature(number_t x, number_t y) {
  //   Reset(x, y);
  //   LOG(INFO) << "feature #" << id_ << " created";
  // }
  void Reset(number_t x, number_t y);

private:
  static int counter_;
  static constexpr int counter0 = 10000;  // feature counter to start with, to guarantee feature id and group id do not coincide  FIXME (xfei): need to ensure group counter does not hit this number
  int id_;
  int sind_; // state index
  FeatureStatus status_;
  GroupPtr ref_; // reference group: where the feature is first observed
  int lifetime_;

  Vec3 x_, x0_; // state: (x, y, inv_z)
  Mat3 P_;      // covariance
  Vec2 pred_;   // predicted pixel coordinates

  Vec3 Xc_; // cached camera coordinates
  Vec3 Xs_; // cached spatial coordinates

  Eigen::Matrix<number_t, 2, kFullSize> J_;
  Vec2 inn_;
  Mat23 Hx_;

  // outlier rejection
  int init_counter_;
  bool inlier_;
  number_t outlier_counter_;

  static JacobianCache cache_; // in-state measurement jacobian cache
  OOSJacobian oos_;            // out-of-state measurement jacobian cache
  int oos_jac_counter_;        // valid OOS jacobian blocks

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
