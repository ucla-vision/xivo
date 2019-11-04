#include "feature.h"
#include "helpers.h"
#include "group.h"

namespace xivo {

int Feature::ComputeOOSJacobian(const std::vector<Observation> &vobs,
                                const Mat3 &Rbc, const Vec3 &Tbc) {

  int num_contraints =
      std::count_if(vobs.begin(), vobs.end(),
                    [](const Observation &obs) { return obs.g->instate(); });

  if (num_contraints > 1) {
    // A constraint should involve at least 2 poses
    cache_.Xs = this->Xs(SE3{SO3{Rbc}, Tbc});
    oos_jac_counter_ = 0;
    for (auto obs : vobs) {
      if (obs.g->instate()) {
        ComputeOOSJacobianInternal(obs, Rbc, Tbc);
      }
    }

    // perform givens elimination
    oos_jac_counter_ = Givens(oos_.inn, oos_.Hx, oos_.Hf, 2 * oos_jac_counter_);
    // std::cout << "feature #" << id_ << " got " << oos_jac_counter_ << " oos
    // jac blocks\n";
  } else {
    oos_jac_counter_ = 0;
  }

  return oos_jac_counter_;
}

void Feature::ComputeOOSJacobianInternal(const Observation &obs,
                                         const Mat3 &Rbc, const Vec3 &Tbc) {

  auto g = obs.g;
  CHECK(g->sind() != -1);

  int goff = kGroupBegin + 6 * obs.g->sind();
  Mat3 Rsb = g->Rsb();
  Mat3 Rsb_t = Rsb.transpose();
  Vec3 Tsb = g->Tsb();
  Mat3 Rbc_t = Rbc.transpose();

  cache_.Xcn = Rbc_t * Rsb_t * (cache_.Xs - Tsb) - Rbc_t * Tbc;
  cache_.dXcn_dXs = Rbc_t * Rsb_t;
  cache_.dXcn_dTsb = -cache_.dXcn_dXs;
  cache_.dXcn_dWsb = Rbc_t * hat(Rsb_t*(cache_.Xs - Tsb));
  cache_.dXcn_dTbc = -Rbc_t;
  cache_.dXcn_dWbc = Rbc_t*hat(Rsb_t * (cache_.Xs - Tsb) - Tbc);

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

} // namespace xivo
