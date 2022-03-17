// IMU class whose intrinsics can be (optionally) estimated.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

#include "Eigen/Core"

#include "alias.h"
#include "component.h"

namespace xivo {

struct IMUState {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mat3 Ca; // accel scaling (diagonal) and misalignment (upper triangle)
  Mat3 Cg; // gyro scaling (diagonal) and misalignment (upper & lower triangle)

  using Tangent = Eigen::Matrix<number_t, 15, 1>;
  void operator+=(const Tangent &dX);

private:
  // make the constructor prive, so ...
  IMUState() : Ca{Mat3::Identity()}, Cg{Mat3::Identity()} {}
  IMUState(const Mat3 &Ca_in, const Mat3 &Cg_in) : Ca{Ca_in}, Cg{Cg_in} {}
  // only its friend IMU can create the IMUState
  friend class IMU;
};

class IMU : public Component<IMU, IMUState> {
public:
  IMU() = default;
  IMU(const Mat3 &Ca, const Mat3 &Cg);

  void UpdateState(const IMUState::Tangent &dX) { X_ += dX; }
  void BackupState() { X_backup_ = X_; }
  void RestoreState() { X_ = X_backup_; }
  const Mat3 &Ca() const { return X_.Ca; }
  const Mat3 &Cg() const { return X_.Cg; }
  const IMUState& State() { return X_; }

private:
  IMUState X_;
  IMUState X_backup_;
};

} // namespace xivo
