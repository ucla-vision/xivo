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

bool DirectLinearTransformSVD(const SE3 &g12, const Vec2 &xc1, const Vec2 &xc2, Vec3 &X) {
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

  X << V(0, 3), V(1, 3), V(2, 3);
  X /= V(3, 3);

  return true;
}

bool DirectLinearTransformAvg(const SE3 &g12, const Vec2 &xc1, const Vec2 &xc2, Vec3 &X) {
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
  X = (xm + xn) / 2.0;

  return true;
}


bool L1Angular(const SE3 &g01, const Vec2 &xc0, const Vec2 &xc1, Vec3 &X, float max_theta_thresh, float beta_thresh) {

  // Initalize the Rotation and Translation Matricies
  Vec3 t01{g01.T()};
  Mat3 R01{g01.R()};
  Mat3 R10{R01.transpose()};
  Vec3 t10{-1 * R01.transpose() * t01};

  // Create homogeneous coordinates
  Vec3 f0{xc0(0), xc0(1), 1.0};
  f0.normalize();
  Vec3 f1{xc1(0), xc1(1), 1.0};
  f1.normalize();

  Vec3 m0{R10 * f0};
  Vec3 m1{f1};

  float a0 = ((m0 / m0.norm()).cross(t10)).norm();
  float a1 = ((m1 / m1.norm()).cross(t10)).norm();

  Vec3 m0_prime;
  Vec3 m1_prime;

  if(a0 <= a1)
  {
    Vec3 n1 = m1.cross(t10);
    Vec3 n1_hat = n1 / n1.norm();
    m0_prime = m0 - (m0.dot(n1_hat)) * n1_hat;
    m1_prime = m1;
  }
  else
  {
    Vec3 n0 = m0.cross(t10);
    Vec3 n0_hat = n0 / n0.norm();
    m0_prime = m0;
    m1_prime = m1 - (m1.dot(n0_hat)) * n0_hat;
  }

  Vec3 Rf0_prime = m0_prime;
  Vec3 f1_prime = m1_prime;

  Vec3 z = f1_prime.cross(Rf0_prime);


  X = ((z.dot(t10.cross(Rf0_prime))) / pow(z.norm(),2)) * f1_prime;

  // Returns point from 1st frame of reference
  X = R01 * X + t01;

  // Check the conditions
  if(!check_cheirality(z, t10, f1_prime, Rf0_prime) ||
    !check_angular_reprojection(m0, Rf0_prime, m1, f1_prime, max_theta_thresh) ||
    !check_parallax(Rf0_prime, f1_prime, beta_thresh))
  {
    return false;
  }

  return true;
}


bool L2Angular(const SE3 &g01, const Vec2 &xc0, const Vec2 &xc1, Vec3 &X, float max_theta_thresh, float beta_thresh) {

  // Initalize the Rotation and Translation Matricies
  Vec3 t01{g01.T()};
  Mat3 R01{g01.R()};
  Mat3 R10{R01.transpose()};
  Vec3 t10{-1 * R01.transpose() * t01};

  // Create homogeneous coordinates
  Vec3 f0{xc0(0), xc0(1), 1.0};
  f0.normalize();
  Vec3 f1{xc1(0), xc1(1), 1.0};
  f1.normalize();

  Vec3 m0{R10 * f0};
  Vec3 m1{f1};

  Vec3 m0_hat = m0 / m0.norm();
  Vec3 m1_hat = m1 / m1.norm();

  Eigen::Matrix<double, 3, 2> A;
  A.row(0) << m0_hat(0), m1_hat(0);
  A.row(1) << m0_hat(1), m1_hat(1);
  A.row(2) << m0_hat(2), m1_hat(2);

  Vec3 t10_hat = t10 / t10.norm();
  Mat3 I = Eigen::Matrix3d::Identity();

  Eigen::Matrix<double, 2, 3> B;
  B = A.transpose() * (I - t10_hat * t10_hat.transpose());

  Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> svd(B, Eigen::ComputeFullV);
  Mat3 V = svd.matrixV();
  Vec3 n_prime_hat = V.col(1);

  Vec3 m0_prime = m0 - m0.dot(n_prime_hat) * n_prime_hat;
  Vec3 m1_prime = m1 - m1.dot(n_prime_hat) * n_prime_hat;

  Vec3 Rf0_prime = m0_prime;
  Vec3 f1_prime = m1_prime;

  Vec3 z = f1_prime.cross(Rf0_prime);

  X = ((z.dot(t10.cross(Rf0_prime))) / pow(z.norm(),2)) * f1_prime;

  // Returns point from 1st frame of reference
  X = R01 * X + t01;

  // Check the conditions
  if(!check_cheirality(z, t10, f1_prime, Rf0_prime) ||
    !check_angular_reprojection(m0, Rf0_prime, m1, f1_prime, max_theta_thresh) ||
    !check_parallax(Rf0_prime, f1_prime, beta_thresh))
  {
    return false;
  }

  return true;
}

bool LinfAngular(const SE3 &g01, const Vec2 &xc0, const Vec2 &xc1, Vec3 &X, float max_theta_thresh, float beta_thresh) {

  // Initalize the Rotation and Translation Matricies
  Vec3 t01{g01.T()};
  Mat3 R01{g01.R()};
  Mat3 R10{R01.transpose()};
  Vec3 t10{-1 * R01.transpose() * t01};

  // Create homogeneous coordinates
  Vec3 f0{xc0(0), xc0(1), 1.0};
  f0.normalize();
  Vec3 f1{xc1(0), xc1(1), 1.0};
  f1.normalize();

  Vec3 m0{R10 * f0};
  Vec3 m1{f1};

  Vec3 m0_hat = m0 / m0.norm();
  Vec3 m1_hat = m1 / m1.norm();

  Vec3 n_a = (m0_hat + m1_hat).cross(t10);
  Vec3 n_b = (m0_hat - m1_hat).cross(t10);

  Vec3 n_prime_hat = n_a.norm() >= n_b.norm() ? n_a : n_b;

  Vec3 m0_prime = m0 - m0.dot(n_prime_hat) * n_prime_hat;
  Vec3 m1_prime = m1 - m1.dot(n_prime_hat) * n_prime_hat;

  Vec3 Rf0_prime = m0_prime;
  Vec3 f1_prime = m1_prime;

  Vec3 z = f1_prime.cross(Rf0_prime);

  X = ((z.dot(t10.cross(Rf0_prime))) / pow(z.norm(),2)) * f1_prime;

  // Returns point from 1st frame of reference
  X = R01 * X + t01;

  // Check the conditions
  if(!check_cheirality(z, t10, f1_prime, Rf0_prime) ||
    !check_angular_reprojection(m0, Rf0_prime, m1, f1_prime, max_theta_thresh) ||
    !check_parallax(Rf0_prime, f1_prime, beta_thresh))
  {
    return false;
  }

  return true;
}


bool check_cheirality(const Vec3 &z, const Vec3 &t, const Vec3 &f1_prime, const Vec3 &Rf0_prime)
{

  float lambda0 = z.dot(t.cross(f1_prime)) / pow(z.norm(), 2);
  float lambda1 = z.dot(t.cross(Rf0_prime)) / pow(z.norm(), 2);

  if(lambda0 <= 0 || lambda1 <= 0)
  {
    LOG(WARNING) << "[WARNING] cheirality error in triangulation. lambda0=" << lambda0 << ", lamba1=" << lambda1;
    return false;
  }

  return true;
}


bool check_angular_reprojection(const Vec3 &Rf0, const Vec3 &Rf0_prime, const Vec3 &f1, const Vec3 &f1_prime, float max_theta_thresh)
{

  float theta0 = acos(Rf0.dot(Rf0_prime) / (Rf0.norm() * Rf0_prime.norm()));
  float theta1 = acos(f1.dot(f1_prime) / (f1.norm() * f1_prime.norm()));

  float max_theta = std::max(theta0, theta1);

  if(max_theta > max_theta_thresh)
  {
    LOG(WARNING) << "[WARNING] angular reprojection error in triangulation";
    return false;
  }
  return true;
}

bool check_parallax(const Vec3 &Rf0_prime, const Vec3 &f1_prime, float beta_thresh)
{

  float beta = acos(f1_prime.dot(Rf0_prime) / (f1_prime.norm() * Rf0_prime.norm()));

  if(beta < beta_thresh)
  {
    LOG(WARNING) << "[WARNING] parallax error in triangulation " << beta;
    return false;
  }

  return true;
}


Mat93 Rsb_Wsb_deriv(const Vec3 Wsb, const Vec3 omega_sb_b) {

  if (Wsb.norm() < 1e-6) {
    return dhat<number_t>() * hat(omega_sb_b);
  }

  number_t Wsb_1 = Wsb(0);
  number_t Wsb_2 = Wsb(1);
  number_t Wsb_3 = Wsb(2);

  number_t omega_sb_b_1 = omega_sb_b(0);
  number_t omega_sb_b_2 = omega_sb_b(1);
  number_t omega_sb_b_3 = omega_sb_b(2);

  number_t A0[9][3];

  number_t t2 = fabs(Wsb_1);
  number_t t3 = fabs(Wsb_2);
  number_t t4 = fabs(Wsb_3);
  number_t t5 = (Wsb_1/fabs(Wsb_1));
  number_t t6 = (Wsb_2/fabs(Wsb_2));
  number_t t7 = (Wsb_3/fabs(Wsb_3));
  number_t t8 = Wsb_1*Wsb_1;
  number_t t9 = Wsb_2*Wsb_2;
  number_t t10 = Wsb_3*Wsb_3;
  number_t t11 = t2*t2;
  number_t t12 = t3*t3;
  number_t t13 = t4*t4;
  number_t t14 = t11+t12+t13;
  number_t t15 = 1.0/t14;
  number_t t17 = sqrt(t14);
  number_t t16 = t15*t15;
  number_t t18 = 1.0/t17;
  number_t t20 = cos(t17);
  number_t t21 = sin(t17);
  number_t t19 = t18*t18*t18;
  number_t t22 = t20-1.0;
  number_t t23 = t18*t21;
  number_t t31 = Wsb_1*t2*t5*t15*t20;
  number_t t32 = Wsb_2*t2*t5*t15*t20;
  number_t t33 = Wsb_1*t3*t6*t15*t20;
  number_t t34 = Wsb_3*t2*t5*t15*t20;
  number_t t35 = Wsb_2*t3*t6*t15*t20;
  number_t t36 = Wsb_1*t4*t7*t15*t20;
  number_t t37 = Wsb_3*t3*t6*t15*t20;
  number_t t38 = Wsb_2*t4*t7*t15*t20;
  number_t t39 = Wsb_3*t4*t7*t15*t20;
  number_t t24 = Wsb_1*t15*t22;
  number_t t25 = Wsb_2*t15*t22;
  number_t t26 = Wsb_3*t15*t22;
  number_t t27 = -t23;
  number_t t46 = -t31;
  number_t t47 = -t32;
  number_t t48 = -t33;
  number_t t49 = -t34;
  number_t t50 = -t35;
  number_t t51 = -t36;
  number_t t52 = -t37;
  number_t t53 = -t38;
  number_t t54 = -t39;
  number_t t55 = Wsb_1*t2*t5*t19*t21;
  number_t t56 = Wsb_2*t2*t5*t19*t21;
  number_t t57 = Wsb_1*t3*t6*t19*t21;
  number_t t58 = Wsb_3*t2*t5*t19*t21;
  number_t t59 = Wsb_2*t3*t6*t19*t21;
  number_t t60 = Wsb_1*t4*t7*t19*t21;
  number_t t61 = Wsb_3*t3*t6*t19*t21;
  number_t t62 = Wsb_2*t4*t7*t19*t21;
  number_t t63 = Wsb_3*t4*t7*t19*t21;
  number_t t73 = t2*t5*t8*t19*t21;
  number_t t74 = t2*t5*t9*t19*t21;
  number_t t75 = t3*t6*t8*t19*t21;
  number_t t76 = t2*t5*t10*t19*t21;
  number_t t77 = t3*t6*t9*t19*t21;
  number_t t78 = t4*t7*t8*t19*t21;
  number_t t79 = t3*t6*t10*t19*t21;
  number_t t80 = t4*t7*t9*t19*t21;
  number_t t81 = t4*t7*t10*t19*t21;
  number_t t91 = Wsb_1*Wsb_2*t2*t5*t16*t22*2.0;
  number_t t92 = Wsb_1*Wsb_3*t2*t5*t16*t22*2.0;
  number_t t93 = Wsb_1*Wsb_2*t3*t6*t16*t22*2.0;
  number_t t94 = Wsb_2*Wsb_3*t2*t5*t16*t22*2.0;
  number_t t95 = Wsb_1*Wsb_3*t3*t6*t16*t22*2.0;
  number_t t96 = Wsb_1*Wsb_2*t4*t7*t16*t22*2.0;
  number_t t97 = Wsb_2*Wsb_3*t3*t6*t16*t22*2.0;
  number_t t98 = Wsb_1*Wsb_3*t4*t7*t16*t22*2.0;
  number_t t99 = Wsb_2*Wsb_3*t4*t7*t16*t22*2.0;
  number_t t100 = t2*t5*t8*t16*t22*2.0;
  number_t t101 = t2*t5*t9*t16*t22*2.0;
  number_t t102 = t3*t6*t8*t16*t22*2.0;
  number_t t103 = t2*t5*t10*t16*t22*2.0;
  number_t t104 = t3*t6*t9*t16*t22*2.0;
  number_t t105 = t4*t7*t8*t16*t22*2.0;
  number_t t106 = t3*t6*t10*t16*t22*2.0;
  number_t t107 = t4*t7*t9*t16*t22*2.0;
  number_t t108 = t4*t7*t10*t16*t22*2.0;
  number_t t28 = t24*2.0;
  number_t t29 = t25*2.0;
  number_t t30 = t26*2.0;
  number_t t40 = -t24;
  number_t t42 = -t25;
  number_t t44 = -t26;
  number_t t64 = Wsb_2*t55;
  number_t t65 = Wsb_3*t55;
  number_t t66 = Wsb_2*t57;
  number_t t67 = Wsb_3*t56;
  number_t t68 = Wsb_3*t57;
  number_t t69 = Wsb_2*t60;
  number_t t70 = Wsb_3*t59;
  number_t t71 = Wsb_3*t60;
  number_t t72 = Wsb_3*t62;
  number_t t82 = -t55;
  number_t t83 = -t56;
  number_t t84 = -t57;
  number_t t85 = -t58;
  number_t t86 = -t59;
  number_t t87 = -t60;
  number_t t88 = -t61;
  number_t t89 = -t62;
  number_t t90 = -t63;
  number_t t109 = t74+t76+t101+t103;
  number_t t110 = t75+t79+t102+t106;
  number_t t111 = t78+t80+t105+t107;
  number_t t41 = -t28;
  number_t t43 = -t29;
  number_t t45 = -t30;
  number_t t112 = t23+t31+t67+t82+t94;
  number_t t113 = t23+t35+t68+t86+t95;
  number_t t114 = t23+t39+t69+t90+t96;
  number_t t115 = t27+t46+t55+t67+t94;
  number_t t116 = t27+t50+t59+t68+t95;
  number_t t117 = t27+t54+t63+t69+t96;
  number_t t118 = t34+t42+t64+t85+t91;
  number_t t119 = t42+t49+t58+t64+t91;
  number_t t120 = t32+t44+t65+t83+t92;
  number_t t121 = t44+t47+t56+t65+t92;
  number_t t122 = t37+t40+t66+t88+t93;
  number_t t123 = t40+t52+t61+t66+t93;
  number_t t124 = t33+t44+t70+t84+t97;
  number_t t125 = t44+t48+t57+t70+t97;
  number_t t126 = t38+t40+t71+t89+t98;
  number_t t127 = t40+t53+t62+t71+t98;
  number_t t128 = t36+t42+t72+t87+t99;
  number_t t129 = t42+t51+t60+t72+t99;
  number_t t130 = t41+t73+t74+t100+t101;
  number_t t131 = t41+t73+t76+t100+t103;
  number_t t132 = t43+t75+t77+t102+t104;
  number_t t133 = t43+t77+t79+t104+t106;
  number_t t134 = t45+t78+t81+t105+t108;
  number_t t135 = t45+t80+t81+t107+t108;
  A0[0][0] = -omega_sb_b_2*t120+omega_sb_b_3*t119;
  A0[0][1] = -omega_sb_b_2*t113+omega_sb_b_3*t123;
  A0[0][2] = omega_sb_b_3*t117-omega_sb_b_2*t126;
  A0[1][0] = omega_sb_b_3*t109+omega_sb_b_1*t120;
  A0[1][1] = omega_sb_b_1*t113+omega_sb_b_3*t133;
  A0[1][2] = omega_sb_b_1*t126+omega_sb_b_3*t135;
  A0[2][0] = -omega_sb_b_2*t109-omega_sb_b_1*t119;
  A0[2][1] = -omega_sb_b_1*t123-omega_sb_b_2*t133;
  A0[2][2] = -omega_sb_b_1*t117-omega_sb_b_2*t135;
  A0[3][0] = -omega_sb_b_2*t115-omega_sb_b_3*t131;
  A0[3][1] = -omega_sb_b_3*t110-omega_sb_b_2*t125;
  A0[3][2] = -omega_sb_b_2*t129-omega_sb_b_3*t134;
  A0[4][0] = omega_sb_b_1*t115-omega_sb_b_3*t118;
  A0[4][1] = -omega_sb_b_3*t122+omega_sb_b_1*t125;
  A0[4][2] = -omega_sb_b_3*t114+omega_sb_b_1*t129;
  A0[5][0] = omega_sb_b_2*t118+omega_sb_b_1*t131;
  A0[5][1] = omega_sb_b_1*t110+omega_sb_b_2*t122;
  A0[5][2] = omega_sb_b_2*t114+omega_sb_b_1*t134;
  A0[6][0] = omega_sb_b_3*t112+omega_sb_b_2*t130;
  A0[6][1] = omega_sb_b_3*t124+omega_sb_b_2*t132;
  A0[6][2] = omega_sb_b_2*t111+omega_sb_b_3*t128;
  A0[7][0] = -omega_sb_b_3*t121-omega_sb_b_1*t130;
  A0[7][1] = -omega_sb_b_3*t116-omega_sb_b_1*t132;
  A0[7][2] = -omega_sb_b_1*t111-omega_sb_b_3*t127;
  A0[8][0] = -omega_sb_b_1*t112+omega_sb_b_2*t121;
  A0[8][1] = omega_sb_b_2*t116-omega_sb_b_1*t124;
  A0[8][2] = -omega_sb_b_1*t128+omega_sb_b_2*t127;

  Mat93 out;
  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < 3; j++) {
      out(i,j) = A0[i][j];
    }
  }

  return out;
}


} // namespace xivo