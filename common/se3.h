// SE(3) groups.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

#include <ostream>
#include "rodrigues.h"

namespace xivo {

namespace lie {

template <typename F> class SO3 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Mat3 = Eigen::Matrix<F, 3, 3>;
  using Vec3 = Eigen::Matrix<F, 3, 1>;

  explicit SO3() : R_{Mat3::Identity()} {}

  // explicit SO3(const Eigen::Matrix<F, 3, 3> &R):R_{R} {}
  template <typename Derived>
  explicit SO3(const Eigen::MatrixBase<Derived> &R) : R_{fitToSO3(R)} {}

  explicit SO3(const Vec3 &axis, F angle)
      : R_{rodrigues(Vec3{axis / axis.norm() * angle})} {}

  SO3 operator*(const SO3 &other) const {
    return SO3{R_ * other.matrix()};
  }
  SO3 &operator*=(const SO3 &other) {
    R_ *= other.matrix();
    return *this;
  }

  template <typename Derived>
  Vec3 operator*(const Eigen::MatrixBase<Derived> &v) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
    return Vec3{R_ * v};
  }

  SO3 inv() const { return SO3{R_.transpose()}; }
  Mat3 matrix() const { return R_; }
  Vec3 log() const { return invrodrigues(R_); }
  static Vec3 log(const SO3 &R) { return R.log(); }
  static SO3 exp(const Vec3 &w) { return SO3{rodrigues(w)}; }

  static Mat3 fitToSO3(const Mat3 &R_approx) {
    Eigen::JacobiSVD<Mat3> svd(R_approx,
                               Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * Mat3::Identity() * svd.matrixV().transpose();
  }

  template <typename TT> SO3<TT> cast() const {
    return SO3(R_.template cast<TT>());
  }
  template <typename TT> static SO3<TT> cast(SO3 &R) {
    return R.cast<TT>();
  }

  // factory methods
  template <typename Derived>
  static SO3 from_matrix(const Eigen::MatrixBase<Derived> &other) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3);
    Eigen::Matrix<F, 3, 3> R = other.template cast<F>();
    return SO3(R);
  }

  template <typename Derived>
  static SO3 project(const Eigen::MatrixBase<Derived> &R) {

    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3);
    using T = typename Derived::Scalar;
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(R, Eigen::ComputeFullU |
                                                        Eigen::ComputeFullV);
    return SO3{svd.matrixU() * Eigen::Matrix<T, 3, 3>::Identity() *
                   svd.matrixV().transpose()};
  }

  // conversion to 3x3 matrix
  operator Eigen::Matrix<F, 3, 3>() const { return R_.matrix(); }

  friend std::ostream& operator<<(std::ostream& os, const SO3& so3) {
    os << so3.matrix();
    return os;
  }

private:
  Mat3 R_;
};

template <typename F> class SE3 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using PointF = Eigen::Matrix<F, 3, 1>;

  explicit SE3() : R_{}, T_{0, 0, 0} {}

  template <typename Derived>
  explicit SE3(const Eigen::MatrixBase<Derived> &RT)
      : R_{RT.template block<3, 3>(0, 0)}, T_{RT.template block<3, 1>(0, 3)} {}

  template <typename Derived>
  explicit SE3(const SO3<F> &R, const Eigen::MatrixBase<Derived> &T)
      : R_{R}, T_{T} {}

  template <typename Derived>
  explicit SE3(const Eigen::MatrixBase<Derived> &R,
                   const Eigen::MatrixBase<Derived> &T)
      : R_{R}, T_{T} {}

  SE3 operator*(const SE3 &other) const {
    return SE3{SO3<F>{R_ * other.so3()},
                   this->so3() * other.translation() + T_};
  }
  SE3 &operator*=(const SE3 &other) {
    R_ *= other.so3();
    T_ += R_ * other.translation();
    return *this;
  }

  template <typename Derived>
  PointF operator*(const Eigen::MatrixBase<Derived> &v) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
    return R_ * v + T_;
  };

  SE3 inv() const { return SE3{R_.inv(), -(R_.inv() * T_)}; }

  SO3<F> so3() const { return R_; }
  SO3<F> R() const { return R_; }
  SO3<F> rotation() const { return R_; }

  SO3<F> &so3() { return R_; }
  SO3<F> &R() { return R_; }
  SO3<F> &rotation() { return R_; }

  PointF translation() const { return T_; }
  PointF T() const { return T_; }

  PointF &translation() { return T_; }
  PointF &T() { return T_; }

  Eigen::Matrix<F, 3, 4> matrix3x4() const {
    return (Eigen::Matrix<F, 3, 4>{} << R_.matrix(), T_).finished();
  }
  operator Eigen::Matrix<F, 3, 4>() const { return matrix3x4(); }

  Eigen::Matrix<F, 4, 4> matrix() const {
    Eigen::Matrix<F, 4, 4> out;
    out.setIdentity();
    out.template block<3, 4>(0, 0) = matrix3x4();
    return out;
  };

  Eigen::Matrix<F, 4, 4> matrix4x4() const { return matrix(); }
  operator Eigen::Matrix<F, 4, 4>() const { return matrix(); }

  template <typename TT> SE3<TT> cast() {
    return SE3<TT>{R_.template cast<TT>(), T_.template cast<TT>()};
  }

  // factory methods
  template <typename Derived>
  static SE3 from_matrix3x4(const Eigen::MatrixBase<Derived> &other) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 4);
    Eigen::Matrix<F, 3, 3> R =
        other.template block<3, 3>(0, 0).template cast<F>();
    Eigen::Matrix<F, 3, 1> T =
        other.template block<3, 1>(0, 3).template cast<F>();
    return SE3{SO3<F>{R}, T};
  }

  template <typename Derived1, typename Derived2>
  static SE3 from_RT(const Eigen::MatrixBase<Derived1> &Rin,
                         const Eigen::MatrixBase<Derived2> &Tin) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived1, 3, 3);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 1);

    Eigen::Matrix<F, 3, 3> R = Rin.template cast<F>();
    Eigen::Matrix<F, 3, 1> T = Tin.template cast<F>();
    return SE3(R, T);
  }

  friend std::ostream& operator<<(std::ostream &os, const SE3& se3) {
    os << se3.matrix3x4();
    return os;
  }

private:
  SO3<F> R_;
  Eigen::Matrix<F, 3, 1> T_;
};

} // namespace lie


// Apply rigid-body transformation to a 3D point.
// Args:
//  R: Rotational part of the transformation.
//  T: Translational part of the transformation.
//  Xin: Inpt 3D point.
//  dX_d[R, T, Xin]: Jacobian matrices w.r.t. R, T, and Xin.
// Returns: 
//  the transformed 3D point Xout
template <typename F>
Eigen::Matrix<F, 3, 1> 
Transform(const lie::SO3<F> &R, 
    const Eigen::Matrix<F, 3, 1> &T, 
    const Eigen::Matrix<F, 3, 1> &Xin,
    Eigen::Matrix<F, 3, 3> *dX_dW=nullptr,
    Eigen::Matrix<F, 3, 3> *dX_dT=nullptr,
    Eigen::Matrix<F, 3, 3> *dX_dXin=nullptr) 
{
  Eigen::Matrix<F, 3, 1> X{R * Xin + T};
  if (dX_dW) {
    *dX_dW = - R.matrix() * hat(Xin);
  }

  if (dX_dT) {
    dX_dT->setIdentity();
  }

  if (dX_dXin) {
    *dX_dXin = R.matrix();
  }
  return X;
}

// Given two poses g1 and g2, compose them as g=g1*g2=(R1*R2, R1*T2 + T1)
// Args:
//  R1, T1: rotational and translational part of pose g1.
//  R2, T2: ... of g2.
//  dW_dW1: differential of the error state W (of R) w.r.t. the error state W1 (of R1)
// Returns:
//  a tuple (R, T) of the composed pose g=(R, T)
template <typename F>
std::tuple<lie::SO3<F>, Eigen::Matrix<F, 3, 1>> 
Compose(
    const lie::SO3<F> &R1, const Eigen::Matrix<F, 3, 1> &T1,
    const lie::SO3<F> &R2, const Eigen::Matrix<F, 3, 1> &T2,
    Eigen::Matrix<F, 3, 3> *dW_dW1=nullptr, Eigen::Matrix<F, 3, 3> *dW_dW2=nullptr,
    Eigen::Matrix<F, 3, 3> *dT_dW1=nullptr, Eigen::Matrix<F, 3, 3> *dT_dT1=nullptr, Eigen::Matrix<F, 3, 3> *dT_dT2=nullptr)
{
  lie::SO3<F> R{R1 * R2};
  Eigen::Matrix<F, 3, 1> T{R1 * T2 + T1};

  if (dW_dW1) {
    *dW_dW1 = R2.matrix().transpose();
  }
  if (dW_dW2) {
    dW_dW2->setIdentity();
  }
  if (dT_dW1) {
    *dT_dW1 = - R1.matrix() * hat(T2);
  }
  if (dT_dT1) {
    dT_dT1->setIdentity();
  }
  if (dT_dT2) {
    *dT_dT2 = R1.matrix();
  }
  return std::make_tuple(R, T);
}

// Given a pose g=(R, T), returns its inverse gi=(R', -R' T)
// Args:
//  R, T: rotational and translational part of input pose g.
//  dWi_dW: differential of the error state Wi (of Ri) w.r.t. the error state W (of R)
//  ...
// Returns:
//  a tuple (Ri, Ti) of inverse pose
template <typename F>
std::tuple<lie::SO3<F>, Eigen::Matrix<F, 3, 1>>
InverseOf(const lie::SO3<F> &R, const Eigen::Matrix<F, 3, 1> &T,
    Eigen::Matrix<F, 3, 3> *dWi_dW=nullptr, 
    Eigen::Matrix<F, 3, 3> *dTi_dW=nullptr,
    Eigen::Matrix<F, 3, 3> *dTi_dT=nullptr) 
{
  lie::SO3<F> Ri{R.inv()};
  Eigen::Matrix<F, 3, 1> Ti{R * -T};

  if (dWi_dW) {
    *dWi_dW = -R.matrix();
  }

  if (dTi_dW) {
    *dTi_dW = hat(Ti);
  }

  if (dTi_dT) {
    *dTi_dT = -R.matrix().transpose();
  }

  return std::make_tuple(R, T);
}


// Given two poses g1 and g2, return the inverse of the composed pose,
// i.e., gi = (g1 g2)^{-1}
// Args:
//  R1, T1: rotational and translational part of the 1st pose g1
//  R2, T2: rotational and translational part of the 2nd pose g2
//  dWi_dW1: differential of error state Wi (of Ri) w.r.t. to error state W1 (of R1)
//  ...
// Returns:
//  A tuple (Ri, Ti), i.e., the rotational and translational part of the inverse composition.
template <typename F>
std::tuple<lie::SO3<F>, Eigen::Matrix<F, 3, 1>> 
InverseOfCompose(
    const lie::SO3<F> &R1, const Eigen::Matrix<F, 3, 1> &T1,
    const lie::SO3<F> &R2, const Eigen::Matrix<F, 3, 1> &T2,
    Eigen::Matrix<F, 3, 3> *dWi_dW1=nullptr, Eigen::Matrix<F, 3, 3> *dWi_dW2=nullptr,
    Eigen::Matrix<F, 3, 3> *dTi_dW1=nullptr, Eigen::Matrix<F, 3, 3> *dTi_dT1=nullptr, Eigen::Matrix<F, 3, 3> *dTi_dT2=nullptr)
{
  Eigen::Matrix<F, 3, 3> dW_dW1, dW_dW2, dT_dW1, dT_dT1, dT_dT2;
  auto [R, T] = Compose(R1, T1, R2, T2, 
      &dW_dW1, &dW_dW2,
      &dT_dW1, &dT_dT1, &dT_dT2);

  Eigen::Matrix<F, 3, 3> dWi_dW, dTi_dW, dTi_dT;
  auto [Ri, Ti] = InverseOf(R, T,
      &dWi_dW, &dTi_dW, &dTi_dT);

  if (dWi_dW1) {
    *dWi_dW1 = dWi_dW * dW_dW1;
  }
  if (dWi_dW2) {
    *dWi_dW2 = dWi_dW * dW_dW2;
  }
  if (dTi_dW1) {
    *dTi_dW1 = dTi_dW * dW_dW1 + dTi_dT * dT_dW1;
  }
  if (dTi_dT1) {
    *dTi_dT1 = dTi_dT * dT_dT1;
  }
  if (dTi_dT2) {
    *dTi_dT2 = dTi_dT * dT_dT2;
  }
  return std::make_tuple(Ri, Ti);
}

}
