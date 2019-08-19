#pragma once
#include "camera.h"

namespace feh {

template <typename T>
class PinholeCamera : public BaseCamera<T, PinholeCamera<T>> {
public:
  using MyBase = BaseCamera<T, PinholeCamera<T>>;
  static constexpr int DIM = 4; // size of intrinsic parameters

  PinholeCamera(int rows, int cols, T fx, T fy, T cx, T cy)
      : BaseCamera<T, PinholeCamera<T>>{rows, cols, fx, fy, cx, cy} {}

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> Project(
      const Eigen::MatrixBase<Derived> &xc,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;
    Eigen::Matrix<f_t, 2, 1> xp{fx_ * xc(0) + cx_, fy_ * xc(1) + cy_};

    if (jac != nullptr) {
      // fill in jacobians
      (*jac) << fx_, 0, 0, fy_;
    }

    if (jacc != nullptr) {
      auto &J{*jacc};
      J.setZero(2, 4); // d[x, y]_d[fx, fy, cx, cy]
      J << xc(0), 0, 1, 0, 0, xc(1), 0, 1;
    }
    return xp;
  }

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> UnProject(
      const Eigen::MatrixBase<Derived> &xp,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;
    using Vec2 = Eigen::Matrix<f_t, 2, 1>;
    Vec2 xc{(xp(0) - cx_) / fx_, (xp(1) - cy_) / fy_};

    if (jac != nullptr) {
      (*jac) << 1 / fx_, 0, 0, 1 / fy_;
    }
    return xc;
  }

  void Print(std::ostream &out) const {
    out << "Pinhole Camera" << std::endl
        << "[rows, cols]=" << rows_ << "," << cols_ << "]" << std::endl
        << "[fx, fy, cx, cy]=[" << fx_ << "," << fy_ << "," << cx_ << "," << cy_
        << "]" << std::endl;
  }

protected:
  using MyBase::rows_;
  using MyBase::cols_;
  using MyBase::fx_;
  using MyBase::fy_;
  using MyBase::cx_;
  using MyBase::cy_;
};

} // namespace feh
