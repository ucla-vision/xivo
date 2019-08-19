// Group structure.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <functional>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "component.h"
#include "core.h"

namespace feh {

struct SO3xR3 {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SO3 Rsb;
  Vec3 Tsb;

  using Tangent = Vec6;
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
  int IncrementLifetime() { lifetime_++; }
  int ResetLifetime() { lifetime_ = 0; }

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
  static int counter_;
  int id_;
  int sind_; // state index
  GroupStatus status_;
  int lifetime_;

  // nominal state
  SO3xR3 X_, X0_;
};

} // namespace feh
