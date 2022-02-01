#include "feature.h"
#include "helpers.h"
#include "group.h"
#include "estimator.h"

namespace xivo {

int Feature::ComputeOOSJacobian(const std::vector<Observation> &vobs,
                                const Mat3 &Rbc, const Vec3 &Tbc,
                                const VecX &error_state) {

  int num_constraints =
      std::count_if(vobs.begin(), vobs.end(),
                    [](const Observation &obs) { return obs.g->instate(); });

// A constraint should involve at least 2 poses
  if (num_constraints >= Estimator::instance()->OOS_update_min_observations()) {
    cache_.Xs = this->Xs(SE3{SO3{Rbc}, Tbc});
    oos_jac_counter_ = 0;
    for (auto obs : vobs) {
      if (obs.g->instate()) {
        ComputeOOSJacobianInternal(obs, Rbc, Tbc, error_state);
      }
    }

    // perform givens elimination
//    oos_jac_counter_ = Givens(oos_.inn, oos_.Hx, oos_.Hf, 2 * oos_jac_counter_);
    MatX A;
    oos_jac_counter_ = SlowGivens(oos_.Hf, oos_.Hx, A);
    oos_.inn = A.transpose() * oos_.inn;
    // std::cout << "feature #" << id_ << " got " << oos_jac_counter_ << " oos
    // jac blocks\n";
  } else {
    oos_jac_counter_ = 0;
  }

  return oos_jac_counter_;
}

void Feature::ComputeOOSJacobianInternal(const Observation &obs,
                                         const Mat3 &Rbc, const Vec3 &Tbc,
                                         const VecX &error_state) {

  auto g = obs.g;
  CHECK(g->sind() != -1);

  int goff = kGroupBegin + 6 * obs.g->sind();
  Mat3 Rsb = g->Rsb();
  Mat3 Rsb_t = Rsb.transpose();
  Vec3 Tsb = g->Tsb();
  Mat3 Rbc_t = Rbc.transpose();
  
  // Get error state
  Vec3 Wsb_err = error_state.segment<3>(Index::Wsb);
  Vec3 Tsb_err = error_state.segment<3>(Index::Tsb);
  Vec3 Wbc_err = error_state.segment<3>(Index::Wbc);
  Vec3 Tbc_err = error_state.segment<3>(Index::Tbc);

  cache_.Xcn = Rbc_t * Rsb_t * (cache_.Xs - Tsb) - Rbc_t * Tbc;
  cache_.dXcn_dXs = Rbc_t * Rsb_t;
  cache_.dXcn_dTsb = -cache_.dXcn_dXs;
  cache_.dXcn_dTbc = -Rbc_t;
  //cache_.dXcn_dWsb = Rbc_t * hat(Rsb_t*(cache_.Xs - Tsb));
  //cache_.dXcn_dWbc = Rbc_t*hat(Rsb_t * (cache_.Xs - Tsb) - Tbc);
  Mat93 dR_dWsb_err, dR_dWbc_err;
  Mat3 R_Wsb_err = rodrigues(Wsb_err, &dR_dWsb_err);
  Mat3 R_Wbc_err = rodrigues(Wbc_err, &dR_dWbc_err);
  for (int i=0; i<3; i++) {
    // Grab matrix from column
    Mat3 dR_dWsb_err_i = unstack(dR_dWsb_err.block<9,1>(0,i));
    Mat3 dR_dWbc_err_i = unstack(dR_dWbc_err.block<9,1>(0,i));
    // Compute derivatives
    Vec3 dXcn_dWbci = dR_dWbc_err_i.transpose() * Rbc_t * (Rsb_t*(cache_.Xs - Tsb) - Tbc);
    Vec3 dXcn_dWsbi = Rbc_t * dR_dWsb_err_i.transpose() * Rsb_t * (cache_.Xs - Tsb);
    // Fill in columns in cache
    cache_.dXcn_dWsb.block<3,1>(0,i) = dXcn_dWsbi;
    cache_.dXcn_dWbc.block<3,1>(0,i) = dXcn_dWbci;
  }

  cache_.xcn = project(cache_.Xcn, &cache_.dxcn_dXcn);

  cache_.xp = Camera::instance()->Project(cache_.xcn, &cache_.dxp_dxcn);

  cache_.dxp_dXcn = cache_.dxp_dxcn * cache_.dxcn_dXcn;

  oos_.inn.segment<2>(2 * oos_jac_counter_) = obs.xp - cache_.xp;

  oos_.Hf.block<2, 3>(2 * oos_jac_counter_, 0) =
      cache_.dxp_dXcn * cache_.dXcn_dXs;

  oos_.Hx.block<2, kFullSize>(2 * oos_jac_counter_, 0).setZero();
  oos_.Hx.block<2, 3>(2 * oos_jac_counter_, goff) =
      cache_.dxp_dXcn * cache_.dXcn_dWsb;
  oos_.Hx.block<2, 3>(2 * oos_jac_counter_, goff + 3) =
      cache_.dxp_dXcn * cache_.dXcn_dTsb;
  oos_.Hx.block<2, 3>(2 * oos_jac_counter_, Index::Wbc) =
      cache_.dxp_dXcn * cache_.dXcn_dWbc;
  oos_.Hx.block<2, 3>(2 * oos_jac_counter_, Index::Tbc) =
      cache_.dxp_dXcn * cache_.dXcn_dTbc;
  ++oos_jac_counter_;
}


void Feature::ComputeLCJacobian(const Obs &obs, const Mat3 &Rbc,
                                const Vec3 &Tbc, const VecX &error_state,
                                int match_counter, MatX &H)
{
  auto g = obs.g;

  int goff = kGroupBegin + 6 * obs.g->sind();
  Mat3 Rsb = g->Rsb();
  Mat3 Rsb_t = Rsb.transpose();
  Vec3 Tsb = g->Tsb();
  Mat3 Rbc_t = Rbc.transpose();
  SE3 gbc(SO3(Rbc), Tbc);
  
  // Get error state
  Vec3 Wsb_err = error_state.segment<3>(Index::Wsb);
  Vec3 Tsb_err = error_state.segment<3>(Index::Tsb);
  Vec3 Wbc_err = error_state.segment<3>(Index::Wbc);
  Vec3 Tbc_err = error_state.segment<3>(Index::Tbc);

  cache_.Xcn = Rbc_t * Rsb_t * (Xs(gbc) - Tsb) - Rbc_t * Tbc;
  cache_.dXcn_dXs = Rbc_t * Rsb_t;
  cache_.dXcn_dTsb = -cache_.dXcn_dXs;
  cache_.dXcn_dTbc = -Rbc_t;
  //cache_.dXcn_dWsb = Rbc_t * hat(Rsb_t*(cache_.Xs - Tsb));
  //cache_.dXcn_dWbc = Rbc_t*hat(Rsb_t * (cache_.Xs - Tsb) - Tbc);
  Mat93 dR_dWsb_err, dR_dWbc_err;
  Mat3 R_Wsb_err = rodrigues(Wsb_err, &dR_dWsb_err);
  Mat3 R_Wbc_err = rodrigues(Wbc_err, &dR_dWbc_err);
  for (int i=0; i<3; i++) {
    // Grab matrix from column
    Mat3 dR_dWsb_err_i = unstack(dR_dWsb_err.block<9,1>(0,i));
    Mat3 dR_dWbc_err_i = unstack(dR_dWbc_err.block<9,1>(0,i));
    // Compute derivatives
    Vec3 dXcn_dWbci = dR_dWbc_err_i.transpose() * Rbc_t * (Rsb_t*(cache_.Xs - Tsb) - Tbc);
    Vec3 dXcn_dWsbi = Rbc_t * dR_dWsb_err_i.transpose() * Rsb_t * (cache_.Xs - Tsb);
    // Fill in columns in cache
    cache_.dXcn_dWsb.block<3,1>(0,i) = dXcn_dWsbi;
    cache_.dXcn_dWbc.block<3,1>(0,i) = dXcn_dWbci;
  }

  cache_.xcn = project(cache_.Xcn, &cache_.dxcn_dXcn);

#ifdef USE_ONLINE_CAMERA_CALIB
  Eigen::Matrix<number_t, 2, -1> jacc;
  cache_.xp = Camera::instance()->Project(cache_.xcn, &cache_.dxp_dxcn, &jacc);
#else
  cache_.xp = Camera::instance()->Project(cache_.xcn, &cache_.dxp_dxcn);
#endif

  cache_.dxp_dXcn = cache_.dxp_dxcn * cache_.dxcn_dXcn;

  int st = 2*match_counter;
  H.block<2, 3>(st, goff) = cache_.dxp_dXcn * cache_.dXcn_dWsb;
  H.block<2, 3>(st, goff + 3) = cache_.dxp_dXcn * cache_.dXcn_dTsb;
  H.block<2, 3>(st, Index::Wbc) = cache_.dxp_dXcn * cache_.dXcn_dWbc;
  H.block<2, 3>(st, Index::Tbc) = cache_.dxp_dXcn * cache_.dXcn_dTbc;

#ifdef USE_ONLINE_CAMERA_CALIB
  int dim{Camera::instance()->dim()};
  H.block(st, kCameraBegin, 2, dim) = jacc;
#endif
}

} // namespace xivo
