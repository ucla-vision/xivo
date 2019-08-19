/* Curiously Recurring Template Pattern (CRTP) implementation
 * of camera models. 
 * Problem is now we have multiple different base camera classes:
 * Each base camera type with a specific derived camera type as 
 * the template parameter is a different type.
 */
#pragma once
#include <ostream>
#include <iostream>
#include "Eigen/Core"

// TODO (xfei): separate model and factory
#include "json/json.h"
#include "utils.h"

namespace feh {

template <typename T, typename D>
class BaseCamera_CRTP {
public:
  BaseCamera_CRTP(int rows, int cols, T fx, T fy, T cx, T cy):
    rows_{rows}, cols_{cols}, 
    fx_{fx}, fy_{fy}, cx_{cx}, cy_{cy}
  {}
  BaseCamera_CRTP& operator=(const BaseCamera_CRTP &) = delete;
  BaseCamera_CRTP(const BaseCamera_CRTP &) = delete;
  ~BaseCamera_CRTP() {
  }

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1>
  Project(const Eigen::MatrixBase<Derived> &xc,
          Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac=nullptr) const {
    return static_cast<D*>(this)->Project(xc, jac);
  }

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1>
  UnProject(const Eigen::MatrixBase<Derived> &xp,
            Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac=nullptr) const {
    return static_cast<D*>(this)->UnProject(xp, jac);
  }

  virtual void Print(std::ostream &out) const = 0;

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  T cx() const { return cx_; }
  T cy() const { return cy_; }
  T fx() const { return fx_; }
  T fy() const { return fy_; }

protected:
  int rows_, cols_;
  T fx_, fy_, cx_, cy_;
};

template <typename T>
class ATANCamera: public BaseCamera_CRTP<T, ATANCamera<T>> {
public:
  ATANCamera(int rows, int cols, T fx, T fy, T cx, T cy, T w):
    BaseCamera_CRTP<T, ATANCamera<T>>{rows, cols, fx, fy, cx, cy},
    w_(w), invw_(1.0 / w), tanw2term_(2.0*std::tan(w * 0.5))
  {}

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1>
  Project(const Eigen::MatrixBase<Derived> &xc,
          Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac=nullptr) const
  {
    std::cout << "atan project" << std::endl;
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;

    Eigen::Matrix<f_t, 2, 1> xp;

    f_t R = xc.norm();
    f_t f;
    if (R < 0.0001 ||  w_ == 0) {
      f = 1;
    } else {
      f = invw_ * atan(tanw2term_ * R) / R;
    }

    //Project through distortion model
    xp(0) = fx_ * f * xc(0) + cx_;
    xp(1) = fy_ * f * xc(1) + cy_;

    if (jac != nullptr) {
      // compute jacobians
      if (R < 0.0001 || w_ == 0) {
        (*jac)(0, 0) = fx_;
        (*jac)(1, 1) = fy_;
      } else {
        // FIXME: optimize computation
        f_t df_dx, df_dy, df_dR;
        f_t a = tanw2term_ * R;
        df_dR = invw_ * ( 1./(1+a*a)*a - std::atan(a) ) / R / R;
        df_dx = df_dR * xc(0) / R;
        df_dy = df_dR * xc(1) / R;

        *jac << fx_ * f + fx_ * xc(0) * df_dx, fx_ * xc(0) * df_dy,
        fy_ * xc(1) * df_dx,           fy_ * f + fy_ * xc(1) * df_dy;
      }
    }
    return xp;
  }

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1>
  UnProject(const Eigen::MatrixBase<Derived> &xp,
            Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac=nullptr) const
  {
    std::cout << "atan un-project" << std::endl;
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);

    using f_t = typename Derived::Scalar;
    Eigen::Matrix<f_t, 2, 1> xc;

    Eigen::Matrix<f_t, 2, 1> tmp((xp(0) - cx_) / fx_, (xp(1) - cy_) / fy_);
    f_t R = tmp.norm();
    f_t RR{w_ == 0 ? R : std::tan(R * w_) / tanw2term_};
    f_t f{R > 0.01 ? RR / R : 1.0};
    xc = f * tmp;

    if (jac != nullptr) {
      if (f == 1) {
        (*jac)(0, 0) = 1.0 / fx_;
        (*jac)(1, 1) = 1.0 / fy_;
      } else {
        f_t df_dR;
        f_t a = std::tan(R * w_);
        df_dR = 1.0 / tanw2term_ * (((1+a*a)*w_*R - a) / R / R);

        f_t df_dx, df_dy;
        df_dx = df_dR * tmp(0) / R / fx_;
        df_dy = df_dR * tmp(1) / R / fy_;

        (*jac) << tmp(0) * df_dx + f / fx_, tmp(0) * df_dy,
        tmp(1) * df_dx,           tmp(1) * df_dy + f / fy_;
      }
    }
    return xc;
  }

  void Print(std::ostream &out) const
  {
    out << "ATAN Camera" << std::endl
        << "[rows, cols]=" << rows_ << "," << cols_ << "]" << std::endl
        << "[fx, fy, cx, cy, w]=["
        << fx_ << "," << fy_ << "," << cx_ << "," << cy_ << "," << w_ << "]" << std::endl;
  }

private:
  using BaseCamera_CRTP<T, ATANCamera<T>>::rows_;
  using BaseCamera_CRTP<T, ATANCamera<T>>::cols_;
  using BaseCamera_CRTP<T, ATANCamera<T>>::fx_;
  using BaseCamera_CRTP<T, ATANCamera<T>>::fy_;
  using BaseCamera_CRTP<T, ATANCamera<T>>::cx_;
  using BaseCamera_CRTP<T, ATANCamera<T>>::cy_;

  T w_, invw_, tanw2term_;
};


  // static 
  // ATANCamera* Create(const Json::Value &cfg)
  // {
  //   auto cam_model = cfg["model"].asString();
  //   int rows = cfg["rows"].asInt();
  //   int cols = cfg["cols"].asInt();
  //   T fx = cfg["fx"].asDouble();
  //   T fy = cfg["fy"].asDouble();
  //   T cx = cfg["cx"].asDouble();
  //   T cy = cfg["cy"].asDouble();
  //   T w = cfg["w"].asDouble();
  //   auto cam = new ATANCamera<T>(
  //       rows, cols,
  //       cols * fx, rows * fy, cols * cx, rows * cy, w);
  //   return cam;
  // }


/*
template <typename T>
class EquidistantCamera: public BaseCamera_CRTP<T> {
public:
  EquidistantCamera( int rows, int cols,
                     T fx, T fy, T cx, T cy,
                     T k0, T k1, T k2, T k3,
                     int max_iter=15):
    BaseCamera_CRTP<T>{rows, cols, fx, fy, cx, cy},
    k0_{k0}, k1_{k1}, k2_{k2}, k3_{k3},
    max_iter_{max_iter} {}


  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> Project(
    const Eigen::MatrixBase<Derived> &xc,
    Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac=nullptr) const
  {
    std::cout << "equidistant project" << std::endl;
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;
    Eigen::Matrix<f_t, 2, 1> xp;

    f_t xy_norm2 = xc.squaredNorm();
    f_t xy_norm= sqrt( xy_norm2 );
    f_t xyz_norm2 = xy_norm2 + 1;

    f_t th = atan2( xy_norm, 1.0 );

    f_t phi = atan2( xc[1], xc[0] );

    f_t th2 = th * th;
    f_t th3 = th2 * th;
    f_t th4 = th3 * th;
    f_t th5 = th3 * th2;
    f_t th6 = th5 * th;
    f_t th7 = th5 * th2;
    f_t th8 = th7 * th;
    f_t th9 = th7 * th2;
    f_t r = th + k0_ * th3 + k1_ * th5 + k2_ * th7 + k3_ * th9;

    f_t cos_phi = cos( phi );
    f_t sin_phi = sin( phi );

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

      f_t dr_dth = 1 + k0_ * 3 * th2 + k1_ * 5 * th4 + k2_ * 7 * th6 + k3_ * 9 * th8;

      f_t du_dx = fx_ * cos_phi * dr_dth * dth_dx - fx_ * r * sin_phi * dphi_dx;
      f_t du_dy = fx_ * cos_phi * dr_dth * dth_dy - fx_ * r * sin_phi * dphi_dy;

      f_t dv_dx = fy_ * sin_phi * dr_dth * dth_dx + fy_ * r * cos_phi * dphi_dx;
      f_t dv_dy = fy_ * sin_phi * dr_dth * dth_dy + fy_ * r * cos_phi * dphi_dy;

      // fill in jacobians
      (*jac) << du_dx, du_dy, dv_dx, dv_dy;
    }
    return xp;
  }

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1>
  UnProject(const Eigen::MatrixBase<Derived> &xp,
            Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac=nullptr) const
  {
    std::cout << "equidistant un-project" << std::endl;
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;
    using Vec2 = Eigen::Matrix<f_t, 2, 1>;
    Vec2 xc;

    f_t xn = xp[0] - cx_;
    f_t yn = xp[1] - cy_;

    f_t b(fx_ * yn), a(fy_ * xn);
    f_t phi = atan2( b, a );
    f_t cos_phi = cos( phi );
    f_t sin_phi = sin( phi );

    f_t rth = xn / ( fx_ * cos_phi );

    f_t th = rth;
    // solve th:
    // th + k0*th**3 + k1*th**5 + k2*th**7 + k3*th**9 = rth

    f_t th2, th3, th4, th6, x0, x1;
    for ( int i = 0; i < max_iter_; i++ ) {
      // f = (th + k0*th**3 + k1*th**5 + k2*th**7 + k3*th**9 - rth)^2
      th2 = th * th;
      th3 = th2 * th;
      th4 = th2 * th2;
      th6 = th4 * th2;
      x0  = k0_ * th3 + k1_ * th4 * th + k2_ * th6 * th + k3_ * th6 * th3 - rth + th;
      x1  = 3 * k0_ * th2 + 5 * k1_ * th4 + 7 * k2_ * th6 + 9 * k3_ * th6 * th2 + 1;
      f_t d   = 2 * x0 * x1;
      f_t d2  = 4 * th * x0 * ( 3 * k0_ + 10 * k1_ * th2 + 21 * k2_ * th4 + 36 * k3_ * th6 ) + 2 * x1 * x1;
      f_t delta = d / d2;
      th -= delta;
    }
    f_t tan_th = tan( th );
    xc[0] = tan_th * cos_phi;
    xc[1] = tan_th * sin_phi;

    if (jac != nullptr) {
      f_t a2b2 = a*a+b*b;
      Vec2 dphi_dxy( -b / a2b2 * fy_,  a / a2b2 * fx_);
      Vec2 dcosphi_dxy(-sin_phi * dphi_dxy);
      Vec2 dsinphi_dxy(cos_phi * dphi_dxy);
      Vec2 drth_dxy(cos_phi-xn*dcosphi_dxy[0], -xn*dcosphi_dxy[1]);
      drth_dxy /= (fx_*cos_phi*cos_phi);

      f_t cos_th(cos(th));
      Vec2 dtanth_dxy(drth_dxy / x1 / (cos_th*cos_th));
      Vec2 doutx_dxy(cos_phi * dtanth_dxy + tan_th * dcosphi_dxy);
      Vec2 douty_dxy(sin_phi * dtanth_dxy + tan_th * dsinphi_dxy);
      (*jac) << doutx_dxy[0], doutx_dxy[1],
      douty_dxy[0], douty_dxy[1];
    }
    return xc;
  }

  void Print(std::ostream &out)
  {
    out << "Equidistant Camera" << std::endl
        << "[rows, cols]=" << rows_ << "," << cols_ << "]" << std::endl
        << "[fx, fy, cx, cy]=["
        << fx_ << "," << fy_ << "," << cx_ << "," << cy_ << "]" << std::endl
        << "[k0, k1, k2, k3]=["
        << k0_ << "," << k1_ << "," << k2_ << "," << k3_ << "]" << std::endl;
  }

private:
  using BaseCamera_CRTP<T>::rows_;
  using BaseCamera_CRTP<T>::cols_;
  using BaseCamera_CRTP<T>::fx_;
  using BaseCamera_CRTP<T>::fy_;
  using BaseCamera_CRTP<T>::cx_;
  using BaseCamera_CRTP<T>::cy_;

  T k0_, k1_, k2_, k3_;
  int max_iter_;
};

  // static
  // EquidistantCamera* Create(const Json::Value &cfg)
  // {
  //   int rows = cfg["rows"].asInt();
  //   int cols = cfg["cols"].asInt();
  //   T fx = cfg["fx"].asDouble();
  //   T fy = cfg["fy"].asDouble();
  //   T cx = cfg["cx"].asDouble();
  //   T cy = cfg["cy"].asDouble();
  //   auto k0123 = GetVectorFromJson<ftype, 4>(cfg, "k0123");
  //   int max_iter = cfg["max_iter"].asInt();

  //   auto cam = new EquidistantCamera<T>(
  //       rows, cols,
  //       fx, fy, cx, cy,
  //       k0123[0], k0123[1], k0123[2], k0123[3], 
  //       max_iter);
  //   return cam;
  // }

template <typename T>
class RadialTangentialCamera: public BaseCamera_CRTP<T> {
public:
  RadialTangentialCamera(int rows, int cols,
                     T fx, T fy, T cx, T cy,
                     T r0, T r1,
                     T k0, T k1, T k2,
                     int max_iter=15):
    BaseCamera_CRTP<T>{rows, cols, fx, fy, cx, cy},
    r0_{r0}, r1_{r1},
    k0_{k0}, k1_{k1}, k2_{k2},
    max_iter_{max_iter} {}

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> Project(
    const Eigen::MatrixBase<Derived> &xc,
    Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac=nullptr) const
  {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;
    using Vec2 = Eigen::Matrix<f_t, 2, 1>;
    Vec2 xp;
    return xp;
  }

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1>
  UnProject(const Eigen::MatrixBase<Derived> &xp,
            Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac=nullptr) const
  {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;
    using Vec2 = Eigen::Matrix<f_t, 2, 1>;
    Vec2 xc;
    // TODO (xfei): implement
    return xc;
  }

  void Print(std::ostream &out)
  {
    // TODO (xfei): implement
  }

private:
  using BaseCamera_CRTP<T>::rows_;
  using BaseCamera_CRTP<T>::cols_;
  using BaseCamera_CRTP<T>::fx_;
  using BaseCamera_CRTP<T>::fy_;
  using BaseCamera_CRTP<T>::cx_;
  using BaseCamera_CRTP<T>::cy_;

  T r0_, r1_;
  T k0_, k1_, k2_;
  int max_iter_;
};


template <typename T, typename Cam>
bool OutOfView(T x, T y, const Cam &cam)
{
  float margin(0.01f);
  return x < cam.cols()*margin
         || x >= cam.cols()*(1-margin)
         || y < cam.rows()*margin
         || y >= cam.rows()*(1-margin);
}

template <typename Derived, typename Cam>
bool OutOfView(const Eigen::MatrixBase<Derived> &xp, const Cam &cam)
{
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
  return OutOfView(xp(0), xp(1), cam);
}


*/

}   // feh

