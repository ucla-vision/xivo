// Prince-Dormand numerical integration
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "estimator.h"

namespace feh {

void Estimator::PrinceDormand(const Vec3 &gyro0, const Vec3 &accel0, ftype dt) {
  // reference 1:
  // http://www.mymathlib.com/c_source/diffeq/embedded_runge_kutta/embedded_prince_dormand_v3_4_5.c
  // reference 2:
  // http://depa.fquim.unam.mx/amyd/archivero/DormandPrince_19856.pdf
  static bool pd_initialized{false}, control_stepsize{false};
  static ftype tolerance, min_scale_factor, max_scale_factor, h, h0;
  static int attempts;

  if (!pd_initialized) {
    auto pd_cfg = cfg_["PrinceDormand"];
    control_stepsize = pd_cfg.get("control_stepsize", false).asBool();
    tolerance = pd_cfg.get("tolerance", 1e-3).asDouble();
    attempts = pd_cfg.get("attempts", 12).asInt();
    min_scale_factor = pd_cfg.get("min_scale_factor", 0.125).asDouble();
    max_scale_factor = pd_cfg.get("max_scale_factor", 4.0).asDouble();
    h = pd_cfg.get("stepsize", 0.002).asDouble();
    h0 = h;
    pd_initialized = true;
  }

  if (control_stepsize) {

    ftype total_step = 0.0, scale = 1.0;

    if (h < 1e-6) {
      h = h0;
    }

    h = std::min(h, dt);

    while (total_step < dt) {
      ftype err = PrinceDormandStep(gyro0 + slope_gyro_ * total_step,
                                    accel0 + slope_accel_ * total_step, h);
      total_step += h;
      if (err == 0.0) {
        scale = max_scale_factor;
      } else {
        scale = 0.8 * sqrt(sqrt(tolerance * h / err));
        scale = std::min(std::max(scale, min_scale_factor),
                         max_scale_factor); // clipping
      }
      std::cout << "err=" << err << " ;h=" << h << " ;s=" << scale << std::endl;

      h *= scale;
      if (total_step < dt) {
        if (total_step + h > dt) {
          h = dt - total_step;
        } else if (total_step + h + 0.5 * h > dt) {
          h = 0.5 * h;
        }
      }
    }
  } else {
    // constant h0
    if (h0 < 0) {
      PrinceDormandStep(gyro0, accel0, dt);
    } else {
      ftype total_step = 0;

      Vec3 gyro{gyro0}, accel{accel0};
      while (total_step < dt) {
        ftype h = h0; // this shadows the static variable h
        if (total_step + h > dt) {
          h = dt - total_step;
        } else if (total_step + h + 0.5 * h > dt) {
          // half step trick
          h = 0.5 * h;
        }
        PrinceDormandStep(gyro, accel, h);
        gyro += slope_gyro_ * h;
        accel += slope_accel_ * h;
        total_step += h;
      }
    }
  }
}

ftype Estimator::PrinceDormandStep(const Vec3 &gyro0, const Vec3 &accel0,
                                   ftype dt) {
  static const ftype r_9 = 1.0 / 9.0;
  static const ftype r_2_9 = 2.0 / 9.0;
  static const ftype r_12 = 1.0 / 12.0;
  static const ftype r_324 = 1.0 / 324.0;
  static const ftype r_330 = 1.0 / 330.0;
  static const ftype r_28 = 1.0 / 28.0;
  static const ftype r_400 = 1.0 / 400.0;

  static State X0;
  static Vec3 K1, K2, K3, K4, K5, K6, K7;
  static MatX FK1, FK2, FK3, FK4, FK5, FK6, FK7;
  static MatX PK1, PK2, PK3, PK4, PK5, PK6, PK7;

  ftype step;
  Eigen::Matrix<ftype, 6, 1> slope;
  slope << slope_gyro_, slope_accel_;
  Eigen::Matrix<ftype, 6, 1> gyro_accel0, gyro_accel;
  gyro_accel0 << gyro0, accel0;

  X0 = X_;
  K1 = X0.Vsb;
  ComputeMotionJacobianAt(X0, gyro_accel0);
  FK1 = F_;
  MatX P0 = P_.block<kMotionSize, kMotionSize>(0, 0);
  PK1 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  X0 = X_;
  step = r_2_9 * dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(X0, r_2_9 * (K1), gyro_accel, step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K2 = X0.Vsb;
  FK2 = F_ + F_ * r_2_9 * (FK1)*dt;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) + r_2_9 * (PK1)*dt;
  PK2 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  X0 = X_;
  step = 3.0 * r_9 * dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(X0, r_12 * (K1 + 3.0 * K2), gyro_accel, step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K3 = X0.Vsb;
  FK3 = F_ + F_ * r_12 * (FK1 + 3.0 * FK2) * dt;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) + r_12 * (PK1 + 3.0 * PK2) * dt;
  PK3 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  X0 = X_;
  step = 5.0 * r_9 * dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(X0, r_324 * (55.0 * K1 - 75.0 * K2 + 200.0 * K3), gyro_accel,
                step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K4 = X0.Vsb;
  FK4 = F_ + F_ * r_324 * (55.0 * FK1 - 75.0 * FK2 + 200.0 * FK3) * dt;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) +
       r_324 * (55.0 * PK1 - 75.0 * PK2 + 200.0 * PK3) * dt;
  PK4 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  X0 = X_;
  step = 6.0 * r_9 * dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(X0, r_330 * (83.0 * K1 - 195.0 * K2 + 305.0 * K3 + 27.0 * K4),
                gyro_accel, step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K5 = X0.Vsb;
  FK5 = F_ +
        F_ * r_330 * (83.0 * FK1 - 195.0 * FK2 + 305.0 * FK3 + 27.0 * FK4) * dt;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) +
       r_330 * (83.0 * PK1 - 195.0 * PK2 + 305.0 * PK3 + 27.0 * PK4) * dt;
  PK5 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  X0 = X_;
  step = dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(
      X0, r_28 * (-19.0 * K1 + 63.0 * K2 + 4.0 * K3 - 108.0 * K4 + 88.0 * K5),
      gyro_accel, step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K6 = X0.Vsb;
  FK6 = F_ +
        F_ * r_28 *
            (-19.0 * FK1 + 63.0 * FK2 + 4.0 * FK3 - 108.0 * FK4 + 88.0 * FK5) *
            dt;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) +
       r_28 *
           (-19.0 * PK1 + 63.0 * PK2 + 4.0 * PK3 - 108.0 * PK4 + 88.0 * PK5) *
           dt;
  PK6 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  X0 = X_;
  step = dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(X0, r_400 * (38.0 * K1 + 240.0 * K3 - 243.0 * K4 + 330.0 * K5 +
                             35.0 * K6),
                gyro_accel, step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K7 = X0.Vsb;
  FK7 = F_ +
        F_ * r_400 * (38.0 * FK1 + 240.0 * FK3 - 243.0 * FK4 + 330.0 * FK5 +
                      35.0 * FK6) *
            dt;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) +
       r_400 *
           (38.0 * PK1 + 240.0 * PK3 - 243.0 * PK4 + 330.0 * PK5 + 35.0 * PK6) *
           dt;
  PK7 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  static MatX K, FK, PK;
  K = 0.0862 * K1 + 0.6660 * K3 - 0.7857 * K4 + 0.9570 * K5 + 0.0965 * K6 -
      0.0200 * K7;
  FK = 0.0862 * FK1 + 0.6660 * FK3 - 0.7857 * FK4 + 0.9570 * FK5 +
       0.0965 * FK6 - 0.0200 * FK7;
  PK = 0.0862 * PK1 + 0.6660 * PK3 - 0.7857 * PK4 + 0.9570 * PK5 +
       0.0965 * PK6 - 0.0200 * PK7;

  // apply the aggregated difference to state
  gyro_accel = gyro_accel0 + slope * dt;
  ComposeMotion(X_, K, gyro_accel, dt);

  F_.setIdentity();
  F_ = F_ + FK * dt;

  P_.block<kMotionSize, kMotionSize>(0, 0) =
      P_.block<kMotionSize, kMotionSize>(0, 0) + PK * dt;
  // update the correlation between motion and structure state
  P_.block<kMotionSize, kFullSize - kMotionSize>(0, kMotionSize) =
      F_ * P_.block<kMotionSize, kFullSize - kMotionSize>(0, kMotionSize);
  P_.block<kFullSize - kMotionSize, kMotionSize>(kMotionSize, 0) =
      P_.block<kFullSize - kMotionSize, kMotionSize>(kMotionSize, 0) *
      F_.transpose();

  static MatX diffK;
  diffK = 0.0002 * (44.0 * K1 - 330.0 * K3 + 891.0 * K4 - 660.0 * K5 -
                    45.0 * K6 + 100.0 * K7);
  return std::max<ftype>(fabs(diffK.minCoeff()), fabs(diffK.maxCoeff()));
}
}
