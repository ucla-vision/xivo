#pragma once
#include "camera.h"

namespace feh {

template <typename T>
class EquidistantCamera : public BaseCamera<T, EquidistantCamera<T>> {
public:
  using MyBase = BaseCamera<T, EquidistantCamera<T>>;
  static constexpr int DIM = 8; // size of intrinsic parameters

  EquidistantCamera(int rows, int cols, T fx, T fy, T cx, T cy, T k0, T k1,
                    T k2, T k3, int max_iter = 15)
      : BaseCamera<T, EquidistantCamera<T>>{rows, cols, fx, fy, cx, cy},
        k0_{k0}, k1_{k1}, k2_{k2}, k3_{k3}, max_iter_{max_iter} {}

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> Project(
      const Eigen::MatrixBase<Derived> &xc,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;
    Eigen::Matrix<f_t, 2, 1> xp;

    f_t xy_norm2 = xc.squaredNorm();
    f_t xy_norm = sqrt(xy_norm2);
    f_t xyz_norm2 = xy_norm2 + 1;

    f_t th = std::atan2(xy_norm, 1.0);

    f_t phi = std::atan2(xc[1], xc[0]);

    f_t th2 = th * th;
    f_t th3 = th2 * th;
    f_t th4 = th3 * th;
    f_t th5 = th3 * th2;
    f_t th6 = th5 * th;
    f_t th7 = th5 * th2;
    f_t th8 = th7 * th;
    f_t th9 = th7 * th2;
    f_t r = th + k0_ * th3 + k1_ * th5 + k2_ * th7 + k3_ * th9;

    f_t cos_phi = std::cos(phi);
    f_t sin_phi = std::sin(phi);

    f_t u = fx_ * r * cos_phi + cx_;
    f_t v = fy_ * r * sin_phi + cy_;

    // fill in xp
    xp[0] = u;
    xp[1] = v;

    if (jac != nullptr) {
      f_t dphi_dx = -xc[1] / xy_norm2;
      f_t dphi_dy = xc[0] / xy_norm2;

      f_t dth_dx = xc[0] / xyz_norm2 / xy_norm;
      f_t dth_dy = xc[1] / xyz_norm2 / xy_norm;

      f_t dr_dth =
          1 + k0_ * 3 * th2 + k1_ * 5 * th4 + k2_ * 7 * th6 + k3_ * 9 * th8;

      f_t du_dx = fx_ * cos_phi * dr_dth * dth_dx - fx_ * r * sin_phi * dphi_dx;
      f_t du_dy = fx_ * cos_phi * dr_dth * dth_dy - fx_ * r * sin_phi * dphi_dy;

      f_t dv_dx = fy_ * sin_phi * dr_dth * dth_dx + fy_ * r * cos_phi * dphi_dx;
      f_t dv_dy = fy_ * sin_phi * dr_dth * dth_dy + fy_ * r * cos_phi * dphi_dy;

      // fill in jacobians
      (*jac) << du_dx, du_dy, dv_dx, dv_dy;
    }

    if (jacc != nullptr) {
      auto &J{*jacc};
      J.setZero(2, 8); // d[x,y]_[fx. fy, cx, cy, k0, k1, k2, k3]

      J(0, 0) = r * cos_phi; // dx_dfx
      J(0, 2) = 1;           // dx_dcx
      J(1, 1) = r * sin_phi; // dy_dfy
      J(1, 3) = 1;           // dy_dcy

      Eigen::Matrix<f_t, 1, 4> dr_dk{th3, th5, th7,
                                     th9}; // dr_d[k0, k1, k2, k3]
      J.template block<1, 4>(0, 4) = fx_ * cos_phi * dr_dk;
      J.template block<1, 4>(1, 4) = fy_ * sin_phi * dr_dk;
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
    Vec2 xc;

    f_t xn = xp[0] - cx_;
    f_t yn = xp[1] - cy_;

    f_t b(fx_ * yn), a(fy_ * xn);
    f_t phi = std::atan2(b, a);
    f_t cos_phi = std::cos(phi);
    f_t sin_phi = std::sin(phi);

    f_t rth = xn / (fx_ * cos_phi);

    f_t th = rth;
    // solve th:
    // th + k0*th**3 + k1*th**5 + k2*th**7 + k3*th**9 = rth

    f_t th2, th3, th4, th6, x0, x1;
    for (int i = 0; i < max_iter_; i++) {
      // f = (th + k0*th**3 + k1*th**5 + k2*th**7 + k3*th**9 - rth)^2
      th2 = th * th;
      th3 = th2 * th;
      th4 = th2 * th2;
      th6 = th4 * th2;
      x0 = k0_ * th3 + k1_ * th4 * th + k2_ * th6 * th + k3_ * th6 * th3 - rth +
           th;
      x1 = 3 * k0_ * th2 + 5 * k1_ * th4 + 7 * k2_ * th6 + 9 * k3_ * th6 * th2 +
           1;
      f_t d = 2 * x0 * x1;
      f_t d2 = 4 * th * x0 * (3 * k0_ + 10 * k1_ * th2 + 21 * k2_ * th4 +
                              36 * k3_ * th6) +
               2 * x1 * x1;
      f_t delta = d / d2;
      th -= delta;
    }
    f_t tan_th = std::tan(th);
    xc[0] = tan_th * cos_phi;
    xc[1] = tan_th * sin_phi;

    if (jac != nullptr) {
      f_t a2b2 = a * a + b * b;
      Vec2 dphi_dxy(-b / a2b2 * fy_, a / a2b2 * fx_);
      Vec2 dcosphi_dxy(-sin_phi * dphi_dxy);
      Vec2 dsinphi_dxy(cos_phi * dphi_dxy);
      Vec2 drth_dxy(cos_phi - xn * dcosphi_dxy[0], -xn * dcosphi_dxy[1]);
      drth_dxy /= (fx_ * cos_phi * cos_phi);

      f_t cos_th(cos(th));
      Vec2 dtanth_dxy(drth_dxy / x1 / (cos_th * cos_th));
      Vec2 doutx_dxy(cos_phi * dtanth_dxy + tan_th * dcosphi_dxy);
      Vec2 douty_dxy(sin_phi * dtanth_dxy + tan_th * dsinphi_dxy);
      (*jac) << doutx_dxy[0], doutx_dxy[1], douty_dxy[0], douty_dxy[1];
    }
    return xc;
  }

  void Print(std::ostream &out) const {
    out << "Equidistant Camera" << std::endl
        << "[rows, cols]=" << rows_ << "," << cols_ << "]" << std::endl
        << "[fx, fy, cx, cy]=[" << fx_ << "," << fy_ << "," << cx_ << "," << cy_
        << "]" << std::endl
        << "[k0, k1, k2, k3]=[" << k0_ << "," << k1_ << "," << k2_ << "," << k3_
        << "]" << std::endl;
  }

protected:
  using MyBase::rows_;
  using MyBase::cols_;
  using MyBase::fx_;
  using MyBase::fy_;
  using MyBase::cx_;
  using MyBase::cy_;

  T k0_, k1_, k2_, k3_;
  int max_iter_;
};

} // namespace feh
