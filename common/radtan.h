#pragma once
#include "camera.h"

namespace feh {

template <typename T>
class RadialTangentialCamera : public BaseCamera<T, RadialTangentialCamera<T>> {
public:
  using MyBase = BaseCamera<T, RadialTangentialCamera<T>>;
  static constexpr int DIM = 9; // size of intrinsic parameters

  RadialTangentialCamera(int rows, int cols, T fx, T fy, T cx, T cy, T p1, T p2,
                         T k1, T k2, T k3, int max_iter = 15)
      : BaseCamera<T, RadialTangentialCamera<T>>{rows, cols, fx, fy, cx, cy},
        p1_{p1}, p2_{p2}, k1_{k1}, k2_{k2}, k3_{k3}, max_iter_{max_iter} {}

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> Project(
      const Eigen::MatrixBase<Derived> &xc,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {

    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);

    using f_t = typename Derived::Scalar;
    // intermediate quantities
    f_t x2 = xc(0) * xc(0);
    f_t y2 = xc(1) * xc(1);
    f_t xy = xc(0) * xc(1);
    f_t r2{x2 + y2};
    f_t r4{r2 * r2};
    f_t r6{r2 * r4};
    // compute the correction for radial distortion
    f_t kr{1.0 + k1_ * r2 + k2_ * r4 + k3_ * r6};
    // compute the correction for tangential distortion
    Eigen::Matrix<f_t, 2, 1> kt{2.0 * p1_ * xy + p2_ * (r2 + 2.0 * x2),
                                2.0 * p2_ * xy + p1_ * (r2 + 2.0 * y2)};
    // apply un-distortion
    Eigen::Matrix<f_t, 2, 1> xk{kr * xc + kt};
    // convert to pixels
    Eigen::Matrix<f_t, 2, 1> xp{xk(0) * fx_ + cx_, xk(1) * fy_ + cy_};

    // optionally compute the jacobian
    if (jac != nullptr) {
      // fill in jacobians
      // dxp_dxc = dxp_dxk * dxk_dxc
      // dxk_dxc = dxk_dkr * dkr_dxc + diag([kr, kr]) + dxk_dkt * dkt_dxc
      Eigen::Matrix<f_t, 2, 2> dxp_dxk;
      dxp_dxk << fx_, 0, 0, fy_;
      Eigen::Matrix<f_t, 2, 1> dxk_dkr{xc};

      // dkr_dx = k1_ * 2 * r * dr_dx + k2_ * 4.0 * r3 * dr_dx + k3_ * 6.0
      // * r5 * dr_dx;
      // dkr_dy = k1_ * 2 * r * dr_dy + k2_ * 4.0 * r3 * dr_dy + k3_ * 6.0
      // * r5 * dr_dy;
      // Since dr_dx = x/r and dr_dy = y / r
      // dkr_dx = k1_ * 2 * x + k2_ * 4.0 * r2 * x + k3_ * 6.0 * r4 * x;
      // = (k1_ * 2 + k2_ * 4 * r2 + k3_ * 6 * r4) * x;
      f_t dkr_dxc_coeff = k1_ * 2 + k2_ * 4 * r2 + k3_ * 6 * r4;
      Eigen::Matrix<f_t, 1, 2> dkr_dxc{dkr_dxc_coeff, dkr_dxc_coeff};

      // kt = [2.0 * p1_ * xy + p2_ * (r2 + 2.0 * x2),
      //   2.0 * p2_ * xy + p1_ * (r2 + 2.0 * y2)]
      Eigen::Matrix<f_t, 2, 2> dkt_dxc;
      dkt_dxc << 2.0 * p1_ * xc(1) + p2_ * (2 * xc(0) + 4.0 * xc(0)),
          2.0 * p1_ * xc(0) + p2_ * (2 * xc(1)),
          2.0 * p2_ * xc(1) + p1_ * (2.0 * xc(0)),
          2.0 * p2_ * xc(0) + p1_ * (2 * xc(1) + 4 * xc(1));

      Eigen::Matrix<f_t, 2, 2> dxk_dxc{dxk_dkr * dkr_dxc + dkt_dxc};
      dxk_dxc(0, 0) += kr;
      dxk_dxc(1, 1) += kr;
      *jac = dxp_dxk * dxk_dxc;
    }

    if (jacc != nullptr) {
      auto &J{*jacc};
      J.setZero(2, DIM); // d[x, y]_d[fx, fy, cx, cy, p1, p2, k1, k2, k3]
      J(0, 0) = xk(0);
      J(0, 2) = 1;
      J(1, 1) = xk(1);
      J(1, 3) = 1;
      Eigen::Matrix<f_t, 2, 2> dxk_dp; // d[x, y]_d[p1, p2]
      dxk_dp << 2.0 * xy, r2 + 2.0 * x2, r2 + 2.0 * y2, 2.0 * xy;

      Eigen::Matrix<f_t, 1, 3> dkr_dk{r2, r4, r6}; // dkr_d[k1, k2, k3]
      Eigen::Matrix<f_t, 2, 3> dxk_dk{xc * dkr_dk};
      // dxk_dk << xc(0) * dkr_dk,
      //        xc(1) * dkr_dk;

      J.template block<1, 5>(0, 4) << fx_ * dxk_dp.row(0), fx_ * dxk_dk.row(0);
      J.template block<1, 5>(1, 4) << fy_ * dxk_dp.row(1), fy_ * dxk_dk.row(1);
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
    out << "RadialTangential Camera" << std::endl
        << "[rows, cols]=" << rows_ << "," << cols_ << "]" << std::endl
        << "[fx, fy, cx, cy]=[" << fx_ << "," << fy_ << "," << cx_ << "," << cy_
        << "]"
        << "[p1, p2]=[" << p1_ << "," << p2_ << "]"
        << "[k0, k1, k2]=[" << k1_ << "," << k2_ << "," << k3_ << "]"
        << std::endl;
  }

protected:
  using MyBase::rows_;
  using MyBase::cols_;
  using MyBase::fx_;
  using MyBase::fy_;
  using MyBase::cx_;
  using MyBase::cy_;
  T p1_, p2_, k1_, k2_, k3_;
  int max_iter_;
};

} // namespace feh
