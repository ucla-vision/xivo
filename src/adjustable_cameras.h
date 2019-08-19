#pragma once

#include "alias.h"
#include "component.h"
#include "equidist.h"

namespace feh {

class A_RadialTangentialCamera
    : public RadialTangentialCamera<ftype>,
      public Component<
          A_RadialTangentialCamera,
          Eigen::Matrix<ftype, RadialTangentialCamera<ftype>::DIM, 1>> {

public:
  using f_t = ftype;
  using MyBase = RadialTangentialCamera<f_t>;
  static constexpr int DIM = MyBase::DIM;

  A_RadialTangentialCamera(int rows, int cols, f_t fx, f_t fy, f_t cx, f_t cy,
                           f_t p1, f_t p2, f_t k1, f_t k2, f_t k3,
                           int max_iter = 15)
      : MyBase{rows, cols, fx, fy, cx, cy, p1, p2, k1, k2, k3, max_iter} {}

  void UpdateState(const Eigen::Matrix<ftype, DIM, 1> &dX) {
    fx_ += dX(0);
    fy_ += dX(1);
    cx_ += dX(2);
    cy_ += dX(3);

    p1_ += dX(4);
    p2_ += dX(5);

    k1_ += dX(6);
    k2_ += dX(7);
    k3_ += dX(8);
  }

protected:
  using MyBase::rows_;
  using MyBase::cols_;

  using MyBase::fx_;
  using MyBase::fy_;
  using MyBase::cx_;
  using MyBase::cy_;

  using MyBase::p1_;
  using MyBase::p2_;

  using MyBase::k1_;
  using MyBase::k2_;
  using MyBase::k3_;
};

class A_PinholeCamera
    : public PinholeCamera<ftype>,
      public Component<A_PinholeCamera,
                       Eigen::Matrix<ftype, PinholeCamera<ftype>::DIM, 1>> {

public:
  using f_t = ftype;
  using MyBase = PinholeCamera<f_t>;
  static constexpr int DIM = MyBase::DIM;

  A_PinholeCamera(int rows, int cols, f_t fx, f_t fy, f_t cx, f_t cy)
      : PinholeCamera<ftype>{rows, cols, fx, fy, cx, cy} {}

  void UpdateState(const Eigen::Matrix<ftype, DIM, 1> &dX) {
    fx_ += dX(0);
    fy_ += dX(1);
    cx_ += dX(2);
    cy_ += dX(3);
  }

protected:
  using MyBase::rows_;
  using MyBase::cols_;
  using MyBase::fx_;
  using MyBase::fy_;
  using MyBase::cx_;
  using MyBase::cy_;
};

class A_ATANCamera
    : public ATANCamera<ftype>,
      public Component<A_ATANCamera,
                       Eigen::Matrix<ftype, ATANCamera<ftype>::DIM, 1>> {

public:
  using f_t = ftype;
  using MyBase = ATANCamera<f_t>;
  static constexpr int DIM = MyBase::DIM;

  A_ATANCamera(int rows, int cols, f_t fx, f_t fy, f_t cx, f_t cy, f_t w)
      : ATANCamera<f_t>{rows, cols, fx, fy, cx, cy, w} {}

  void UpdateState(const Eigen::Matrix<ftype, DIM, 1> &dX) {
    fx_ += dX(0);
    fy_ += dX(1);
    cx_ += dX(2);
    cy_ += dX(3);
    w_ += dX(4);
    invw_ = 1.0 / w_;
    w2_ = 2.0 * std::tan(w_ * 0.5);
  }

protected:
  using MyBase::rows_;
  using MyBase::cols_;

  using MyBase::fx_;
  using MyBase::fy_;
  using MyBase::cx_;
  using MyBase::cy_;

  using MyBase::w_;
  using MyBase::invw_;
  using MyBase::w2_;
};

class A_EquidistantCamera
    : public EquidistantCamera<ftype>,
      public Component<A_EquidistantCamera,
                       Eigen::Matrix<ftype, EquidistantCamera<ftype>::DIM, 1>> {

public:
  using f_t = ftype;
  using MyBase = EquidistantCamera<f_t>;
  static constexpr int DIM = MyBase::DIM;

  A_EquidistantCamera(int rows, int cols, f_t fx, f_t fy, f_t cx, f_t cy,
                      f_t k0, f_t k1, f_t k2, f_t k3, int max_iter = 15)
      : EquidistantCamera<f_t>{rows, cols, fx, fy, cx, cy, k0, k1, k2, k3} {}

  void UpdateState(const Eigen::Matrix<ftype, DIM, 1> &dX) {
    fx_ += dX(0);
    fy_ += dX(1);
    cx_ += dX(2);
    cy_ += dX(3);

    k0_ += dX(4);
    k1_ += dX(5);
    k2_ += dX(6);
    k3_ += dX(7);
  }

private:
  using MyBase::rows_;
  using MyBase::cols_;

  using MyBase::fx_;
  using MyBase::fy_;
  using MyBase::cx_;
  using MyBase::cy_;

  using MyBase::k0_;
  using MyBase::k1_;
  using MyBase::k2_;
  using MyBase::k3_;
};

} // namespace feh
