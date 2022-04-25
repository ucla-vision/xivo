// Radial-Tangential camera model.
// Reference: OpenCV documentation.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "camera_base.h"

namespace xivo {

template <typename T>
class RadialTangentialCamera : public BaseCamera<T, RadialTangentialCamera<T>> {
public:
  using MyBase = BaseCamera<T, RadialTangentialCamera<T>>;
  static constexpr int DIM = 9; // size of intrinsic parameters

  RadialTangentialCamera(int rows, int cols, T fx, T fy, T cx, T cy, T p1, T p2,
                         T k1, T k2, T k3, int max_iter = 15)
      : BaseCamera<T, RadialTangentialCamera<T>>{rows, cols, fx, fy, cx, cy},
        p1_{p1}, p2_{p2}, k1_{k1}, k2_{k2}, k3_{k3}, max_iter_{max_iter},
        p10_{0.0}, p20_{0.0}, k10_{0.0}, k20_{0.0}, k30_{0.0} {}

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> Project(
      const Eigen::MatrixBase<Derived> &xc,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {

    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);

    using f_t = typename Derived::Scalar;

    f_t x = xc(0);
    f_t y = xc(1);

    f_t t2 = x*x;
    f_t t3 = y*y;
    f_t t4 = k1_*x*2.0;
    f_t t5 = k1_*y*2.0;
    f_t t6 = p1_*x*2.0;
    f_t t7 = p2_*y*2.0;
    f_t t8 = t2*3.0;
    f_t t9 = t3*3.0;
    f_t t10 = t6*y;
    f_t t11 = t7*x;
    f_t t12 = t2+t3;
    f_t t13 = t3+t8;
    f_t t14 = t2+t9;
    f_t t15 = t12*t12;
    f_t t16 = t12*t12*t12;
    f_t t17 = k1_*t12;
    f_t t22 = k2_*t12*x*4.0;
    f_t t23 = k2_*t12*y*4.0;
    f_t t18 = k2_*t15;
    f_t t19 = k3_*t16;
    f_t t20 = p1_*t14;
    f_t t21 = p2_*t13;
    f_t t24 = k3_*t15*x*6.0;
    f_t t25 = k3_*t15*y*6.0;
    f_t t26 = t4+t22+t24;
    f_t t27 = t5+t23+t25;
    f_t t28 = t17+t18+t19+1.0;
    f_t t29 = t28*x;
    f_t t30 = t28*y;
    f_t t31 = t10+t21+t29;
    f_t t32 = t11+t20+t30;

    Eigen::Matrix<f_t, 2, 1> xp;
    xp(0) = cx_+fx_*t31;
    xp(1) = cy_+fy_*t32;

    if (jac != nullptr) {
      auto &J(*jac);
      J(0,0) = fx_*(t28+p2_*x*6.0+p1_*y*2.0+t26*x);
      J(0,1) = fx_*(t6+t7+t27*x);
      J(1,0) = fy_*(t6+t7+t26*y);
      J(1,1) = fy_*(t28+p2_*x*2.0+p1_*y*6.0+t27*y);
    }

    if (jacc != nullptr) {
      auto &J{*jacc};
      J.setZero(2, DIM); // d[x, y]_d[fx, fy, cx, cy, p1, p2, k1, k2, k3]
      J(0,0) = t31; // fx
      J(0,2) = 1.0; // cx
      J(0,4) = fx_*x*y*2.0; // p1
      J(0,5) = fx_*t13; // p2
      J(0,6) = fx_*t12*x; // k1
      J(0,7) = fx_*t15*x; // k2
      J(0,8) = fx_*t16*x; // k3
      J(1,1) = t32; // fy
      J(1,3) = 1.0; // cy
      J(1,4) = fy_*t14; // p1
      J(1,5) = fy_*x*y*2.0;  // p2
      J(1,6) = fy_*t12*y;    // k1
      J(1,7) = fy_*t15*y;    // k2
      J(1,8) = fy_*t16*y;    // k3
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
    Vec2 xk{(xp(0) - cx_) / fx_, (xp(1) - cy_) / fy_};

    // Initial guess of undistorted point
    Vec2 xc{xk};

    // Use Newton's method to solve for the undistorted point
    for (int i=0; i<max_iter_; i++) {

      f_t x = xc(0);
      f_t y = xc(1);

      f_t t2 = x*x;
      f_t t3 = y*y;
      f_t t4 = k1_*x*2.0;
      f_t t5 = k1_*y*2.0;
      f_t t6 = p1_*x*2.0;
      f_t t7 = p2_*y*2.0;
      f_t t8 = t2+t3;
      f_t t9 = t8*t8;
      f_t t10 = t8*t8*t8;
      f_t t11 = k1_*t8;
      f_t t14 = k2_*t8*x*4.0;
      f_t t15 = k2_*t8*y*4.0;
      f_t t12 = k2_*t9;
      f_t t13 = k3_*t10;
      f_t t16 = k3_*t9*x*6.0;
      f_t t17 = k3_*t9*y*6.0;
      f_t t18 = t4+t14+t16;
      f_t t19 = t5+t15+t17;
      f_t t20 = t11+t12+t13+1.0;

      Eigen::Matrix<f_t, 2, 1> f;
      Eigen::Matrix<f_t, 2, 2> grad_F;

      f_t xkk = t20*x+t6*y+p2_*(t2*2.0+t8);
      f_t ykk = t7*x+t20*y+p1_*(t3*2.0+t8);

      f_t dxk_dx = t20+p2_*x*6.0+p1_*y*2.0+t18*x;
      f_t dxk_dy = t6+t7+t19*x;
      f_t dyk_dx = t6+t7+t18*y;
      f_t dyk_dy = t20+p2_*x*2.0+p1_*y*6.0+t19*y;

      f(0) = xkk - xk(0);
      f(1) = ykk - xk(1);

      grad_F(0,0) = dxk_dx;
      grad_F(0,1) = dxk_dy;
      grad_F(1,0) = dyk_dx;
      grad_F(1,1) = dyk_dy;

      xc << xc - (grad_F.inverse())*f;
    }

    // This part is not implemented properly. The model below is for the
    // pinhole distortion model. It does not work for the radtan model.
    // But, it's currently not used, and I'm not going to fix it (yet).
    if (jac != nullptr) {
      (*jac) << 1 / fx_, 0, 0, 1 / fy_;
    }

    return xc;
  }

  void Print(std::ostream &out) const {
    out << "Radial-Tangential Camera" << std::endl
        << "[rows, cols]=" << rows_ << "," << cols_ << "]" << std::endl
        << "[fx, fy, cx, cy]=[" << fx_ << "," << fy_ << "," << cx_ << "," << cy_
        << "]" << std::endl
        << "[p1, p2]=[" << p1_ << "," << p2_ << "]" << std::endl
        << "[k0, k1, k2]=[" << k1_ << "," << k2_ << "," << k3_ << "]"
        << std::endl;
  }

  Eigen::Matrix<T, 9, 1> GetIntrinsics() {
    Eigen::Matrix<T, 9, 1> output;
    output << fx_, fy_, cx_, cy_, p1_, p2_, k1_, k2_, k3_;
    return output;
  }

  DistortionType GetDistortionType() { return DistortionType::RADTAN; }

  void BackupState() {
    MyBase::BackupState();
    p10_ = p1_;
    p20_ = p2_;
    k10_ = k1_;
    k20_ = k2_;
    k30_ = k3_;
  }

  void RestoreState() {
    MyBase::RestoreState();
    p1_ = p10_;
    p2_ = p20_;
    k1_ = k10_;
    k2_ = k20_;
    k3_ = k30_;
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

  // backup states
  T p10_, p20_, k10_, k20_, k30_;
};

} // namespace xivo
