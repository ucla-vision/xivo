// Jacobian management and cache.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "Eigen/Dense"
#include "core.h"
#include <array>

namespace xivo {

struct OOSJacobian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OOSJacobian() {
    Hx.resize(2 * kMaxGroup, kFullSize);
    Hf.resize(2 * kMaxGroup, 3);
    inn.resize(2 * kMaxGroup);
  }
  MatX Hx; // ... w.r.t. state
  MatX Hf;
  VecX inn; // n is std of measurement noise
};

using OOSJacobianPtr = OOSJacobian *;

struct JacobianCache {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 Xc;     // 3D point in camera frame of the reference group
  Mat3 dXc_dx; // 3D point in reference camera frame w.r.t. local state

  Vec3 Xbr;
  Mat3 dXbr_dTbc, dXbr_dWbc;
  Mat3 dXbr_dXc;

  Vec3 Xs;                 // 3D point in spatial frame
  Mat3 dXs_dTsbr, dXs_dWsbr;   // w.r.t. body2spatial pose of the reference group
  Mat3 dXs_dXbr;            // 3D point in spatial frame w.r.t. Xc

  Vec3 Xb;
  Mat3 dXb_dTsb, dXb_dWsb;
  Mat3 dXb_dXs;

  Vec3 Xcn;      // 3D point in camera frame of the "new" (current) group
  Mat3 dXcn_dTbc, dXcn_dWbc; // w.r.t. cam2body alignment
  Mat3 dXcn_dXb;

  // Chain rule values
  Mat3 dXcn_dTsb, dXcn_dWsb;
  Mat3 dXcn_dTsbr, dXcn_dWsbr;
  Mat3 dXcn_dXs, dXcn_dx;

  Vec3 dXcn_dtd;                       // w.r.t. temporal offset
  Eigen::Matrix<number_t, 3, 9> dXcn_dCg; // w.r.t. gyroscope intrinsics
  Mat3 dXcn_dbg;

  Vec2 xcn;        // camera coordinates in the "new" group
  Mat23 dxcn_dXcn; // w.r.t. 3D point in camera frame

  Vec2 xp; // pixel coordinates
  Mat23 dxp_dXcn;
  Mat2 dxp_dxcn; // w.r.t. new camera coordinates
  Vec2 inn;      // innovation
};

} // namespace xivo
