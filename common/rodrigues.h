#pragma once
#include "Eigen/Dense"
#include "math.h"
#include <type_traits>

namespace xivo {

constexpr int Sum1ToN(int N) { return (N * (N + 1)) >> 1; }

template <typename Derived, int N = 3>
Eigen::Matrix<typename Derived::Scalar, N * N, Sum1ToN(N)>
dA_dAu(const Eigen::MatrixBase<Derived> &A) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, N, N);

  Eigen::Matrix<typename Derived::Scalar, N * N, Sum1ToN(N)> D;
  int idx_u{0}; // indexing to the upper triangular part of the matrix
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      D(i * N + j, idx_u++) = 1;
    }
  }
  return D;
}

template <typename T, int N = 3>
Eigen::Matrix<T, N * N, Sum1ToN(N)>
dA_dAu() {
  Eigen::Matrix<T, N * N, Sum1ToN(N)> D;
  int idx_u{0}; // indexing to the upper triangular part of the matrix
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      D(i * N + j, idx_u++) = 1;
    }
  }
  return D;
}

template <typename Derived, int M = 3, int N = 3>
Eigen::Matrix<typename Derived::Scalar, M, N>
unstack(const Eigen::MatrixBase<Derived> &u, int major = Eigen::RowMajor) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, M * N, 1);
  Eigen::Matrix<typename Derived::Scalar, M, N> m;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      if (major == Eigen::RowMajor) {
        m(i, j) = u(i * M + j);
      } else {
        m(j, i) = u(i * M + j);
      }
    }
  return m;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
hat(const Eigen::MatrixBase<Derived> &u) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  return (Eigen::Matrix<typename Derived::Scalar, 3, 3>{} << 0, -u(2), u(1),
          u(2), 0, -u(0), -u(1), u(0), 0)
      .finished();
}

template <typename T = float> Eigen::Matrix<T, 9, 3> dhat() {
  return (Eigen::Matrix<T, 9, 3>{} << 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 0, 0,
          0, -1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0)
      .finished();
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 9, 3>
dhat(const Eigen::MatrixBase<Derived> &u) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  return dhat<typename Derived::Scalar>();
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1>
vee(const Eigen::MatrixBase<Derived> &R) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3);
  return {R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1)};
}

template <typename T = float> Eigen::Matrix<T, 3, 9> dvee() {
  return (Eigen::Matrix<T, 3, 9>{} << 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 0, 0,
          0, -1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0)
      .finished();
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 9>
dvee(const Eigen::MatrixBase<Derived> &R) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3);
  return dvee<typename Derived::Scalar>();
}

template <typename T = float, int N = 3, int M = 3>
Eigen::Matrix<T, M * N, N * M> dAt_dA() {
  Eigen::Matrix<T, M * N, N * M> D;
  D.setZero();
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      D(m * N + n, n * M + m) = 1;
    }
  }
  return D;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar,
              Derived::RowsAtCompileTime * Derived::ColsAtCompileTime,
              Derived::RowsAtCompileTime * Derived::ColsAtCompileTime>
dAt_dA(const Eigen::MatrixBase<Derived> &A) {
  return dAt_dA<typename Derived::Scalar, Derived::RowsAtCompileTime,
                Derived::ColsAtCompileTime>();
}

template <int RowA, int ColA, typename Derived>
Eigen::Matrix<typename Derived::Scalar,
              RowA * Derived::ColsAtCompileTime,
              RowA * ColA>
dAB_dA(const Eigen::MatrixBase<Derived> &B) {

  using T = typename Derived::Scalar;
  constexpr int N = RowA;
  constexpr int M = ColA;
  constexpr int P = Derived::ColsAtCompileTime;

  static_assert(M == Derived::RowsAtCompileTime,
                "Columns of A should match rows of B.");

  Eigen::Matrix<T, N * P, N * M> D;
  D.setZero();
  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < P; ++p) {
      for (int m = 0; m < M; ++m) {
        D(n * P + p, n * M + m) += B(m, p);
      }
    }
  }
  return D;
}

// Note, by default the Eigen matrices arrange data in RowMajor order.
// This does not affect the way we index the element via () operator.
// But when using Map<> function to map raw internal data to matrices/vectors,
// we need to be careful about the order.
// dC_{n,p}/dA_{n,m}=B_{m,p}
template <typename Derived, typename OtherDerived>
Eigen::Matrix<typename Derived::Scalar,
              Derived::RowsAtCompileTime * OtherDerived::ColsAtCompileTime,
              Derived::RowsAtCompileTime * Derived::ColsAtCompileTime>
dAB_dA(const Eigen::MatrixBase<Derived> &A,
       const Eigen::MatrixBase<OtherDerived> &B) {

  return dAB_dA<Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>(B);

  /*
  using T = typename Derived::Scalar;
  constexpr int N = Derived::RowsAtCompileTime;
  constexpr int M = Derived::ColsAtCompileTime;
  constexpr int P = OtherDerived::ColsAtCompileTime;

  static_assert(std::is_same<T, typename OtherDerived::Scalar>::value,
                "Operands should have same dtype.");
  static_assert(M == OtherDerived::RowsAtCompileTime,
                "Columns of A should match rows of B.");

  Eigen::Matrix<T, N * P, N * M> D;
  D.setZero();
  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < P; ++p) {
      for (int m = 0; m < M; ++m) {
        D(n * P + p, n * M + m) += B(m, p);
      }
    }
  }
  return D;
  */
}


template <int RowB, int ColB, typename Derived>
Eigen::Matrix<typename Derived::Scalar,
              Derived::RowsAtCompileTime * ColB,
              RowB * ColB>
dAB_dB(const Eigen::MatrixBase<Derived> &A) {

  using T = typename Derived::Scalar;
  constexpr int N = Derived::RowsAtCompileTime;
  constexpr int M = Derived::ColsAtCompileTime;
  constexpr int P = ColB;
  static_assert(M == Derived::RowsAtCompileTime,
                "Columns of A should match rows of B.");

  Eigen::Matrix<T, N * P, M * P> D;
  D.setZero();
  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < P; ++p) {
      for (int m = 0; m < M; ++m) {
        D(n * P + p, m * P + p) += A(n, m);
      }
    }
  }
  return D;
}



// dC_{n,p}/dB_{m,p}=A_{n,m}
template <typename Derived, typename OtherDerived>
Eigen::Matrix<typename Derived::Scalar,
              Derived::RowsAtCompileTime * OtherDerived::ColsAtCompileTime,
              OtherDerived::RowsAtCompileTime * OtherDerived::ColsAtCompileTime>
dAB_dB(const Eigen::MatrixBase<Derived> &A,
       const Eigen::MatrixBase<OtherDerived> &B) {

  return dAB_dB<OtherDerived::RowsAtCompileTime, OtherDerived::ColsAtCompileTime>(A);

  /*
  using T = typename Derived::Scalar;
  constexpr int N = Derived::RowsAtCompileTime;
  constexpr int M = Derived::ColsAtCompileTime;
  constexpr int P = OtherDerived::ColsAtCompileTime;
  static_assert(std::is_same<T, typename OtherDerived::Scalar>::value,
                "Operands should have same type.");
  static_assert(M == OtherDerived::RowsAtCompileTime,
                "Columns of A should match rows of B.");

  Eigen::Matrix<T, N * P, M * P> D;
  D.setZero();
  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < P; ++p) {
      for (int m = 0; m < M; ++m) {
        D(n * P + p, m * P + p) += A(n, m);
      }
    }
  }
  return D;
  */
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
rodrigues(const Eigen::MatrixBase<Derived> &w,
          Eigen::Matrix<typename Derived::Scalar, 9, 3> *dR_dw = nullptr) {

  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  using T = typename Derived::Scalar;
  Eigen::Matrix<T, 3, 3> R;

  T th = w.norm();

  if (th < 1e-8) {
    // R = I + hat(w)
    // std::cout << "small angle approximation" << std::endl;
    R = Eigen::Matrix<T, 3, 3>::Identity() + hat(w);
    if (dR_dw) {
      *dR_dw = dhat(w);
    }
    return R;
  }
  T inv_th = 1.0 / th;
  Eigen::Matrix<T, 3, 1> u = w * inv_th;

  // R = I + u.hat * sin(th) + (u.hat)^2 * (1-cos(th))
  T sin_th = sin(th);
  T cos_th = cos(th);
  Eigen::Matrix<T, 3, 3> uhat = hat(u);
  Eigen::Matrix<T, 3, 3> uhat2 = uhat * uhat;
  R = Eigen::Matrix<T, 3, 3>::Identity() + uhat * sin_th + uhat2 * (1 - cos_th);
  if (dR_dw) {
    Eigen::Matrix<T, 9, 3> dR_du =
        sin_th * dhat(u) +
        (1 - cos_th) * (dAB_dA<3, 3>(uhat) + dAB_dB<3, 3>(uhat)) * dhat(u);
    Eigen::Matrix<T, 3, 3> du_dw =
        inv_th * (Eigen::Matrix<T, 3, 3>::Identity() - u * u.transpose());
    Eigen::Matrix<T, 9, 1> dR_dth = Eigen::Map<Eigen::Matrix<T, 9, 1>>(
        Eigen::Matrix<T, 3, 3>{uhat * cos_th + uhat2 * sin_th}.data());
    // Eigen::Matrix<T, 1, 3> dth_dw = u.transpose();
    *dR_dw = dR_du * du_dw + dR_dth * u.transpose();
  }
  return R;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1>
invrodrigues(const Eigen::MatrixBase<Derived> &R,
             Eigen::Matrix<typename Derived::Scalar, 3, 9> *dw_dR = nullptr) {

  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3);
  using T = typename Derived::Scalar;

  Eigen::Matrix<T, 3, 1> w;

  T tmp = 0.5 * (R.trace() - 1);
  Eigen::Matrix<T, 3, 1> vee_R = vee(R);
  if (tmp > 1.0 - 1e-10) {
    // std::cout << "small angle approximation" << std::endl;
    w = 0.5 * vee_R;
    if (dw_dR) {
      *dw_dR = 0.5 * dvee(R);
    }
    return w;
  }

  T th = acos(tmp);
  T sin_th = sin(th);
  T inv_sin_th = 1.0 / sin_th;
  Eigen::Matrix<T, 3, 1> u = 0.5 * vee_R * inv_sin_th;

  w = th * u;

  if (dw_dR) {
    Eigen::Matrix<T, 1, 9> dth_dR;
    T dth_dtmp = -1 / sqrt(1 - tmp * tmp);
    Eigen::Matrix<T, 1, 9> dtmp_dR;
    dtmp_dR << 1, 0, 0, 0, 1, 0, 0, 0, 1; // d(trace(R)-1)_dR
    dtmp_dR *= 0.5;
    dth_dR = dth_dtmp * dtmp_dR;

    Eigen::Matrix<T, 3, 9> du_dR;
    // u = vee(R) / (2*sin(th));
    du_dR = 0.5 * (dvee(R) * inv_sin_th -
                   vee(R) * cos(th) * inv_sin_th * inv_sin_th * dth_dR);
    *dw_dR = u * dth_dR + th * du_dR;
  }
  return w;
}

}
