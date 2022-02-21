#include "helpers.h"

#include <algorithm>
#include <iostream>
#include <memory>

#include "glog/logging.h"

namespace xivo {

// Reference:
// http://www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf
int SlowGivens(const MatX &Hf, MatX &Hx, MatX &A) {
  // FIXME: use Givens rotation
  Eigen::FullPivLU<MatX> lu(Hf.transpose());
  A = lu.kernel();
//  MatX A = lu.kernel(); // Hf.T * A = 0 -> A.T * Hf = 0
  // Hf: 2nx3, Hf.T: 3x2n, A: 2nx?, At: ?x2n
  // MatX At = A.transpose();
  Hx = A.transpose() * Hx;

  return Hx.rows();
}

// Matrix computation. Golub & Loan.
// page 240, Algorithm 5.1.3
static Mat2 givens(number_t a, number_t b) {
  Mat2 G;
  number_t c, s, t;
  if (fabs(b) < eps) {
    c = 1;
    s = 0;
  } else {
    if (fabs(b) > fabs(a)) {
      t = -a / b;
      s = 1 / sqrt(1 + t * t);
      c = s * t;
    } else {
      t = -b / a;
      c = 1 / sqrt(1 + t * t);
      s = c * t;
    }
  }
  G << c, s, -s, c;
  return G;
}

int Givens(VecX &x, MatX &Hx, MatX &Hf, int effective_rows) {
  CHECK((effective_rows == -1 && (x.rows() ^ 1)) ||
        ((effective_rows <= x.rows()) && effective_rows ^ 1));

  CHECK(x.rows() == Hx.rows());
  CHECK(x.rows() == Hf.rows());

  int rows = (effective_rows == -1 ? Hf.rows() : effective_rows);
  int cols = Hf.cols();

  Mat2 Gt;
  for (int c = 0; c < cols; ++c) {
    for (int r = rows - 2; r >= c; --r) {
      Gt.transpose() = givens(Hf(r, c), Hf(r + 1, c));
      Hf.block(r, 0, 2, cols) = Gt * Hf.block(r, 0, 2, cols);
      Hx.block(r, 0, 2, cols) = Gt * Hx.block(r, 0, 2, cols);

      x.segment<2>(r) = Gt * x.segment<2>(r);
    }
  }
  // now strip the first #cols rows
  for (int r = 0; r < rows - cols; ++r) {
    x(r) = x(r + cols);
    Hx.row(r) = Hx.row(r + cols);
    Hf.row(r) = Hf.row(r + cols);
  }
  return rows - cols;
}

int QR(VecX &x, MatX &Hx, int effective_rows) {
  CHECK(x.rows() == Hx.rows());

  int rows = (effective_rows == -1 ? Hx.rows() : effective_rows);
  int cols = Hx.cols();

  CHECK(rows > cols);

  Mat2 Gt;
  for (int c = 0; c < cols; ++c) {
    for (int r = rows - 2; r >= c; --r) {
      Gt.transpose() = givens(Hx(r, c), Hx(r + 1, c));
      Hx.block(r, 0, 2, cols) = Gt * Hx.block(r, 0, 2, cols);
      x.segment<2>(r) = Gt * x.segment<2>(r);
    }
  }
  // in contrast to the Givens function used to eliminate measurement jacobian
  // Hf of feature,
  // here we want to keep the parts which are not eliminated -- which are the
  // first #rows
  // of both residual vector x and measurement jacobian Hx
  // return the number of rows, and let the caller to decide what to do with the
  // matrices.
  return rows;
}

Vec3 Triangulate1(const SE3 &g12, const Vec2 &xc1, const Vec2 &xc2) {
  Vec3 t12{g12.T()};
  Mat3 R12{g12.R()};
  Mat34 P1;
  P1.block<3, 3>(0, 0).setIdentity();
  Mat34 P2;
  P2.block<3, 3>(0, 0) = R12.transpose();
  P2.block<3, 1>(0, 3) = -R12.transpose() * t12;
  Vec3 f1{xc1(0), xc1(1), 1.0};
  f1.normalize();
  Vec3 f2{xc2(0), xc2(1), 1.0};
  f2.normalize();

  Mat4 A;
  A.row(0) = f1(0) * P1.row(2) - f1(2) * P1.row(0);
  A.row(1) = f1(1) * P1.row(2) - f1(2) * P1.row(1);
  A.row(2) = f2(0) * P2.row(2) - f2(2) * P2.row(0);
  A.row(3) = f2(1) * P2.row(2) - f2(2) * P2.row(1);

  Eigen::JacobiSVD<Mat4> svd(A, Eigen::ComputeFullV);
  auto V = svd.matrixV();
  Vec3 x;
  x << V(0, 3), V(1, 3), V(2, 3);
  x /= V(3, 3);
  return x;
}

Vec3 Triangulate2(const SE3 &g12, const Vec2 &xc1, const Vec2 &xc2) {
  Vec3 t12{g12.T()};
  Mat3 R12{g12.R()};

  Vec3 f1{xc1(0), xc1(1), 1.0};
  f1.normalize();
  Vec3 f2{xc2(0), xc2(1), 1.0};
  f2.normalize();

  Vec3 f2_unrotated{R12 * f2};
  Vec2 b;
  b << t12.dot(f1), t12.dot(f2_unrotated);
  Mat2 A;
  A(0, 0) = f1.dot(f1);
  A(1, 0) = f1.dot(f2_unrotated);
  A(0, 1) = -A(1, 0);
  A(1, 1) = -f2_unrotated.dot(f2_unrotated);
  Vec2 lambda = A.inverse() * b;
  Vec3 xm = lambda(0) * f1;
  Vec3 xn = t12 + lambda(1) * f2_unrotated;
  Vec3 x = (xm + xn) / 2.0; // in frame 1
  return x;
}


Vec3 Triangulate3(const SE3 &g12, const Vec2 &xc1, const Vec2 &xc2) {

  // Initalize the Rotation and Translation Matricies
  Vec3 t12{g12.T()};
  Mat3 R12{g12.R()};
  Mat3 R21{R12.transpose()};
  Vec3 t21{-1 * R12.transpose() * t12};

  // Create homogeneous coordinates
  Vec3 f0{xc1(0), xc1(1), 1.0};
  f0.normalize(); // WHY?
  Vec3 f1{xc2(0), xc2(1), 1.0};
  f1.normalize();  

  Vec3 m0{R21 * f0};
  Vec3 m1{f1};

  float a0 = ((m0 / m0.norm()).cross(t21)).norm();
  float a1 = ((m1 / m1.norm()).cross(t21)).norm();

  Vec3 m0_prime;
  Vec3 m1_prime;

  if(a0 <= a1)
  {
    Vec3 n1 = m1.cross(t21);
    Vec3 n1_hat = n1 / n1.norm();
    m0_prime = m0 - (m0.dot(n1_hat)) * n1_hat;
    m1_prime = m1;
  }
  else
  {
    Vec3 n0 = m0.cross(t21);
    Vec3 n0_hat = n0 / n0.norm();
    m0_prime = m0;
    m1_prime = m1 - (m1.dot(n0_hat)) * n0_hat;
  }

  Vec3 z = m1_prime.cross(m0_prime);

  Vec3 x = ((z.dot(t21.cross(m0_prime))) / pow(z.norm(),2)) * m1_prime;

  return x;
}


Vec3 Triangulate4(const SE3 &g12, const Vec2 &xc1, const Vec2 &xc2) {

  // Initalize the Rotation and Translation Matricies
  Vec3 t12{g12.T()};
  Mat3 R12{g12.R()};
  Mat3 R21{R12.transpose()};
  Vec3 t21{-1 * R12.transpose() * t12};

  // Create homogeneous coordinates
  Vec3 f0{xc1(0), xc1(1), 1.0};
  f0.normalize(); // WHY?
  Vec3 f1{xc2(0), xc2(1), 1.0};
  f1.normalize();  

  Vec3 m0{R21 * f0};
  Vec3 m1{f1};

  Vec3 m0_hat = m0 / m0.norm();
  Vec3 m1_hat = m1 / m1.norm();

  Eigen::Matrix<double, 3, 2> A;
  A.row(0) << m0_hat(0), m1_hat(0);
  A.row(1) << m0_hat(1), m1_hat(1);
  A.row(2) << m0_hat(2), m1_hat(2);

  Vec3 t21_hat = t21 / t21.norm();
  Mat3 I = Eigen::Matrix3d::Identity();

  Eigen::Matrix<double, 2, 3> B;
  B = A.transpose() * (I - t21_hat * t21_hat.transpose());

  Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> svd(B, Eigen::ComputeFullV);
  Mat3 V = svd.matrixV();
  Vec3 n_prime_hat = V.col(1);

  Vec3 m0_prime = m0 - m0.dot(n_prime_hat) * n_prime_hat;
  Vec3 m1_prime = m1 - m1.dot(n_prime_hat) * n_prime_hat;

  Vec3 z = m1_prime.cross(m0_prime);

  Vec3 x = ((z.dot(t21.cross(m0_prime))) / pow(z.norm(),2)) * m1_prime;

  return x;
}

} // namespace xivo
