// Project and Unproject functions.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

#include "Eigen/Core"

namespace xivo {

/** Perspective projection. For input Xc=(X, Y, Z), returns xc=(X/Z, Y/Z).
 *  Optionally computes the 2x3 Jacobian d(xc)/d(Xc) */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 2, 1>
project(const Eigen::MatrixBase<Derived> &Xc,
        Eigen::Matrix<typename Derived::Scalar, 2, 3> *dxc_dXc = nullptr) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);

  Eigen::Matrix<typename Derived::Scalar, 2, 1> xc;
  xc << Xc.template head<2>() / Xc(2);
  if (dxc_dXc) {
    *dxc_dXc << 1 / Xc(2), 0, -Xc(0) / (Xc(2) * Xc(2)), 0, 1 / Xc(2),
        -Xc(1) / (Xc(2) * Xc(2));
  }
  return xc;
}


/** Perspective projection with inverse depth appended.
 *  For input Xc=(X, Y, Z), returns xc=(X/Z, Y/Z, 1/Z).
 *  Optionally computes the 3x3 Jacobian d(xc)/d(Xc) */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1>
project_invz(const Eigen::MatrixBase<Derived> &Xc,
             Eigen::Matrix<typename Derived::Scalar, 3, 3> *dxc_dXc = nullptr) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);

  Eigen::Matrix<typename Derived::Scalar, 3, 1> xc;
  xc << Xc(0) / Xc(2), Xc(1) / Xc(2), 1.0 / Xc(2);
  if (dxc_dXc) {
    *dxc_dXc << 1 / Xc(2), 0, -Xc(0) / (Xc(2) * Xc(2)), 0, 1 / Xc(2),
        -Xc(1) / (Xc(2) * Xc(2)), 0, 0, -1 / (Xc(2) * Xc(2));
  }
  return xc;
}


/** For a projected point xc=(X/Z, Y/Z, 1/Z), returns (X, Y, Z), the
 *  coordinates of the point with respect to the camera frame.
 *  Optionally computes the 3x3 Jacobian d(Xc)/d(xc) */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> unproject_invz(
    const Eigen::MatrixBase<Derived> &xc,
    Eigen::Matrix<typename Derived::Scalar, 3, 3> *dXc_dxc = nullptr) {
  return project_invz(xc, dXc_dxc);
}


/** Perspective projection with log depth appended.
 *  For input Xc=(X, Y, Z), returns xc=(X/Z, Y/Z, log(Z)).
 *  Optionally computes the 3x3 Jacobian d(xc)/d(Xc) */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1>
project_logz(const Eigen::MatrixBase<Derived> &Xc,
             Eigen::Matrix<typename Derived::Scalar, 3, 3> *dxc_dXc = nullptr) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);

  Eigen::Matrix<typename Derived::Scalar, 3, 1> xc;
  xc << Xc(0) / Xc(2), Xc(1) / Xc(2), log(Xc(2));
  if (dxc_dXc) {
    *dxc_dXc << 1 / Xc(2), 0, -Xc(0) / (Xc(2) * Xc(2)), 0, 1 / Xc(2),
        -Xc(1) / (Xc(2) * Xc(2)), 0, 0, 1 / Xc(2);
  }
  return xc;
}


/** For a projected point xc=(X/Z, Y/Z, log(Z)), returns (X, Y, Z), the
 *  coordinates of the point with respect to the camera frame.
 *  Optionally computes the 3x3 Jacobian d(Xc)/d(xc) */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> unproject_logz(
    const Eigen::MatrixBase<Derived> &xc,
    Eigen::Matrix<typename Derived::Scalar, 3, 3> *dXc_dxc = nullptr) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);

  using f_t = typename Derived::Scalar;
  f_t z = exp(xc(2));

  Eigen::Matrix<f_t, 3, 1> Xc;
  Xc << xc(0) * z, xc(1) * z, z;

  if (dXc_dxc) {
    *dXc_dxc << z, 0, xc(0) * z, 0, z, xc(1) * z, 0, 0, z;
  }
  return Xc;
}

} // namespace xivo
