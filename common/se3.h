// SE(3) groups.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

#include "rodrigues.h"

namespace feh {

template <typename Type> class SO3Type {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Mat3 = Eigen::Matrix<Type, 3, 3>;
  using Vec3 = Eigen::Matrix<Type, 3, 1>;

  explicit SO3Type() : R_{Mat3::Identity()} {}

  // explicit SO3Type(const Eigen::Matrix<Type, 3, 3> &R):R_{R} {}
  template <typename Derived>
  explicit SO3Type(const Eigen::MatrixBase<Derived> &R) : R_{R} {}
  explicit SO3Type(const Vec3 &axis, Type angle)
      : R_{rodrigues(Vec3{axis / axis.norm() * angle})} {}

  SO3Type operator*(const SO3Type &other) const {
    return SO3Type{R_ * other.matrix()};
  }
  SO3Type &operator*=(const SO3Type &other) {
    R_ *= other.matrix();
    return *this;
  }

  template <typename Derived>
  Vec3 operator*(const Eigen::MatrixBase<Derived> &v) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
    return Vec3{R_ * v};
  }

  SO3Type inv() const { return SO3Type{R_.transpose()}; }
  Mat3 matrix() const { return R_; }
  Vec3 log() const { return invrodrigues(R_); }
  static Vec3 log(const SO3Type &R) { return R.log(); }
  static SO3Type exp(const Vec3 &w) { return SO3Type{rodrigues(w)}; }

  static SO3Type fitToSO3(const Mat3 &R_approx) {
    Eigen::JacobiSVD<Mat3> svd(R_approx,
                               Eigen::ComputeThinU | Eigen::ComputeThinV);
    return SO3Type{svd.matrixU() * Mat3::Identity() *
                   svd.matrixV().transpose()};
  }

  template <typename TT> SO3Type<TT> cast() const {
    return SO3Type(R_.template cast<TT>());
  }
  template <typename TT> static SO3Type<TT> cast(SO3Type &R) {
    return R.cast<TT>();
  }

  // factory methods
  template <typename Derived>
  static SO3Type from_matrix(const Eigen::MatrixBase<Derived> &other) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3);
    Eigen::Matrix<Type, 3, 3> R = other.template cast<Type>();
    return SO3Type(R);
  }

  template <typename Derived>
  static SO3Type project(const Eigen::MatrixBase<Derived> &R) {

    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3);
    using T = typename Derived::Scalar;
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(R, Eigen::ComputeFullU |
                                                        Eigen::ComputeFullV);
    return SO3Type{svd.matrixU() * Eigen::Matrix<T, 3, 3>::Identity() *
                   svd.matrixV().transpose()};
  }

  // conversion to 3x3 matrix
  operator Eigen::Matrix<Type, 3, 3>() const { return R_.matrix(); }

private:
  Mat3 R_;
};

template <typename Type> class SE3Type {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using PointType = Eigen::Matrix<Type, 3, 1>;

  explicit SE3Type() : R_{}, T_{0, 0, 0} {}

  template <typename Derived>
  explicit SE3Type(const Eigen::MatrixBase<Derived> &RT)
      : R_{RT.template block<3, 3>(0, 0)}, T_{RT.template block<3, 1>(0, 3)} {}

  template <typename Derived>
  explicit SE3Type(const SO3Type<Type> &R, const Eigen::MatrixBase<Derived> &T)
      : R_{R}, T_{T} {}

  template <typename Derived>
  explicit SE3Type(const Eigen::MatrixBase<Derived> &R,
                   const Eigen::MatrixBase<Derived> &T)
      : R_{R}, T_{T} {}

  SE3Type operator*(const SE3Type &other) const {
    return SE3Type{SO3Type<Type>{R_ * other.so3()},
                   this->so3() * other.translation() + T_};
  }
  SE3Type &operator*=(const SE3Type &other) {
    R_ *= other.so3();
    T_ += R_ * other.translation();
    return *this;
  }

  template <typename Derived>
  PointType operator*(const Eigen::MatrixBase<Derived> &v) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
    return R_ * v + T_;
  };

  SE3Type inv() const { return SE3Type{R_.inv(), -(R_.inv() * T_)}; }

  SO3Type<Type> so3() const { return R_; }
  SO3Type<Type> R() const { return R_; }
  SO3Type<Type> rotation() const { return R_; }

  SO3Type<Type> &so3() { return R_; }
  SO3Type<Type> &R() { return R_; }
  SO3Type<Type> &rotation() { return R_; }

  PointType translation() const { return T_; }
  PointType T() const { return T_; }

  PointType &translation() { return T_; }
  PointType &T() { return T_; }

  Eigen::Matrix<Type, 3, 4> matrix3x4() const {
    return (Eigen::Matrix<Type, 3, 4>{} << R_.matrix(), T_).finished();
  }
  operator Eigen::Matrix<Type, 3, 4>() const { return matrix3x4(); }

  Eigen::Matrix<Type, 4, 4> matrix() const {
    Eigen::Matrix<Type, 4, 4> out;
    out.setIdentity();
    out.template block<3, 4>(0, 0) = matrix3x4();
    return out;
  };

  Eigen::Matrix<Type, 4, 4> matrix4x4() const { return matrix(); }
  operator Eigen::Matrix<Type, 4, 4>() const { return matrix(); }

  template <typename TT> SE3Type<TT> cast() {
    return SE3Type<TT>{R_.template cast<TT>(), T_.template cast<TT>()};
  }

  // factory methods
  template <typename Derived>
  static SE3Type from_matrix3x4(const Eigen::MatrixBase<Derived> &other) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 4);
    Eigen::Matrix<Type, 3, 3> R =
        other.template block<3, 3>(0, 0).template cast<Type>();
    Eigen::Matrix<Type, 3, 1> T =
        other.template block<3, 1>(0, 3).template cast<Type>();
    return SE3Type{SO3Type<Type>{R}, T};
  }

  template <typename Derived1, typename Derived2>
  static SE3Type from_RT(const Eigen::MatrixBase<Derived1> &Rin,
                         const Eigen::MatrixBase<Derived2> &Tin) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived1, 3, 3);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 1);

    Eigen::Matrix<Type, 3, 3> R = Rin.template cast<Type>();
    Eigen::Matrix<Type, 3, 1> T = Tin.template cast<Type>();
    return SE3Type(R, T);
  }

private:
  SO3Type<Type> R_;
  Eigen::Matrix<Type, 3, 1> T_;
};
}
