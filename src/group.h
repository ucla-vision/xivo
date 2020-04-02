// Group structure which are anchors to which
// the features are attached.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <functional>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "component.h"
#include "core.h"

namespace xivo {

struct SO3xR3 {
  static constexpr int DIM = 6;
  using Tangent = Eigen::Matrix<number_t, DIM, 1>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SO3 Rsb;
  Vec3 Tsb;

  SO3xR3 &operator+=(const Tangent &dX) {
    Rsb *= SO3::exp(dX.head<3>());
    Tsb += dX.tail<3>();
    return *this;
  }
};

class Group : public Component<Group, SO3xR3> {
  friend class MemoryManager;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static GroupPtr Create(const SO3 &Rsb, const Vec3 &Tsb);
  static void Delete(GroupPtr g);

  // id & index related accessors
  int id() const { return id_; }
  int sind() const { return sind_; }
  void SetSind(int ind) { sind_ = ind; }
  int lifetime() const { return lifetime_; }
  void IncrementLifetime() { lifetime_++; }
  void ResetLifetime() { lifetime_ = 0; }

  void BackupState() { X0_ = X_; }
  void RestoreState() { X_ = X0_; }

  // status accessors
  bool instate() const;
  GroupStatus status() const { return status_; }
  void SetStatus(GroupStatus status) { status_ = status; }

  // local state accessors
  SE3 gsb() const { return SE3{X_.Rsb, X_.Tsb}; }
  const SO3 &Rsb() const { return X_.Rsb; }
  const Vec3 &Tsb() const { return X_.Tsb; }
  void SetState(const SE3 &gsb) { SetState(gsb.R(), gsb.T()); }
  void SetState(const SO3 &Rsb, const Vec3 &Tsb) {
    X_.Rsb = Rsb;
    X_.Tsb = Tsb;
  }
  void UpdateState(const Vec6 &dX) { X_ += dX; }

private:
  Group(const Group &) = delete;
  // default constructor used in memory manager's pre-allocation
  Group() = default;
  void Reset(const SO3 &Rsb, const Vec3 &Tsb);

private:

  /** Total number of groups created */
  static int counter_;

  /** ID of group - IDs are in order of group creation and not reused */
  int id_;

  /** Group's slot index in the Estimator's array of groups. This is set only when
   *  the group is added to the Estimator's state and not when the group is first
   *  created. */
  int sind_;

  /** Newly created, Instate, Floating, or Gauge */
  GroupStatus status_;

  /** Number of images that have been processed since group was created. */
  int lifetime_;

  /** Nominal State: (Rsb, Tsb) */
  SO3xR3 X_;

  /** Backup of `X_` = (Rsb, Tsb) used in `Estimator::OnePointRANSAC` */
  SO3xR3 X0_;
};

} // namespace xivo
