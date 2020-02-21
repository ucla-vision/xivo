// ATAN camera model.
// Reference:
//  Straight lines have to be straight ...
//  https://hal.inria.fr/inria-00267247/file/distcalib.pdf
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "camera_base.h"

namespace xivo {

template <typename T> class ATANCamera : public BaseCamera<T, ATANCamera<T>> {
public:
  using MyBase = BaseCamera<T, ATANCamera<T>>;
  static constexpr int DIM = 5; // size of intrinsic parameters

  ATANCamera(int rows, int cols, T fx, T fy, T cx, T cy, T w)
      : BaseCamera<T, ATANCamera<T>>{rows, cols, fx, fy, cx, cy}, w_(w),
        invw_(1.0 / w), w2_(2.0 * std::tan(w * 0.5)) {}

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> Project(
      const Eigen::MatrixBase<Derived> &xc,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;

    Eigen::Matrix<f_t, 2, 1> xp;

    f_t R = xc.norm();
    f_t f{1};
    bool singular = (R < 0.0001 || w_ == 0);

    if (!singular) {
      f = invw_ * std::atan(w2_ * R) / R;
    }

    // Project through distortion model
    xp(0) = fx_ * f * xc(0) + cx_;
    xp(1) = fy_ * f * xc(1) + cy_;

    if (jac != nullptr) {
      auto &J{*jac};
      // compute jacobians
      if (singular) {
        J(0, 0) = fx_;
        J(1, 1) = fy_;
      } else {
        // FIXME: optimize computation
        f_t df_dx, df_dy, df_dR;
        f_t a = w2_ * R;
        df_dR = invw_ * (1. / (1 + a * a) * a - std::atan(a)) / R / R;
        df_dx = df_dR * xc(0) / R;
        df_dy = df_dR * xc(1) / R;

        J << fx_ * f + fx_ * xc(0) * df_dx, fx_ * xc(0) * df_dy,
            fy_ * xc(1) * df_dx, fy_ * f + fy_ * xc(1) * df_dy;
      }
    }

    if (jacc != nullptr) {
      auto &J{*jacc};
      J.setZero(2, DIM); // d[x, y]_d[fx, fy, cx, cy, w]
      if (singular) {
        J << xc(0), 0, 1, 0, 0, 0, xc(1), 0, 1, 0;
      } else {
        J(0, 0) = f * xc(0);
        J(0, 2) = 1;
        J(1, 1) = f * xc(1);
        J(1, 3) = 1;
        // f = inv(w) * atan(w2 * R) / R
        // R is constant w.r.t. w
        f_t df_dinvw = std::atan(w2_ * R) / R;
        f_t dinvw_dw = -invw_ * invw_;

        f_t df_datanw2R = invw_ / R;
        // datan(x)_dx = 1 / (1 + x * x)
        f_t datanw2R_dw2R = 1 / (1 + (w2_ * R) * (w2_ * R));
        f_t dw2R_dw2 = R; // w2R = w2 * R
        // Recall: w2 = 2 * tan(w * 0.5)
        // dw2_dw = 2.0 / cos^2(w*0.5) * 0.5 = 1.0 / cos^2(2 * 0.5);
        f_t dw2_dw = 1 / std::cos(w_ * 0.5);
        dw2_dw *= dw2_dw;
        f_t df_dw = df_dinvw * dinvw_dw +
                    df_datanw2R * datanw2R_dw2R * dw2R_dw2 * dw2_dw;
        J(0, 4) = fx_ * xc(0) * df_dw;
        J(1, 4) = fy_ * xc(1) * df_dw;
      }
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
    Eigen::Matrix<f_t, 2, 1> xc;

    Eigen::Matrix<f_t, 2, 1> tmp((xp(0) - cx_) / fx_, (xp(1) - cy_) / fy_);
    f_t R = tmp.norm();
    f_t RR{w_ == 0 ? R : std::tan(R * w_) / w2_};
    f_t f{R > 0.01 ? RR / R : 1.0};
    xc = f * tmp;

    if (jac != nullptr) {
      if (f == 1) {
        (*jac)(0, 0) = 1.0 / fx_;
        (*jac)(1, 1) = 1.0 / fy_;
      } else {
        f_t df_dR;
        f_t a = std::tan(R * w_);
        df_dR = 1.0 / w2_ * (((1 + a * a) * w_ * R - a) / R / R);

        f_t df_dx, df_dy;
        df_dx = df_dR * tmp(0) / R / fx_;
        df_dy = df_dR * tmp(1) / R / fy_;

        (*jac) << tmp(0) * df_dx + f / fx_, tmp(0) * df_dy, tmp(1) * df_dx,
            tmp(1) * df_dy + f / fy_;
      }
    }
    return xc;
  }

  void Print(std::ostream &out) const {
    out << "ATAN Camera" << std::endl
        << "[rows, cols]=" << rows_ << "," << cols_ << "]" << std::endl
        << "[fx, fy, cx, cy, w]=[" << fx_ << "," << fy_ << "," << cx_ << ","
        << cy_ << "," << w_ << "]" << std::endl;
  }

  Eigen::Matrix<T, 9, 1> GetIntrinsics() {
    Eigen::Matrix<T, 9, 1> output;
    output << fx_, fy_, cx_, cy_, w_, 0, 0, 0, 0;
    return output;
  }

  DistortionType GetDistortionType() { return DistortionType::ATAN; }


protected:
  using MyBase::rows_;
  using MyBase::cols_;
  using MyBase::fx_;
  using MyBase::fy_;
  using MyBase::cx_;
  using MyBase::cy_;

  T w_, invw_, w2_;
};

} // namespace xivo
