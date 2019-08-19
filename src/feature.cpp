#include <algorithm>

#include "estimator.h"
#include "feature.h"
#include "group.h"
#include "helpers.h"
#include "mm.h"
#include "param.h"

#include "Eigen/QR"
#include "absl/strings/str_format.h"
#include "glog/logging.h"

namespace feh {

// Feature
int Feature::counter_ = 0;
JacobianCache Feature::cache_ = {};

////////////////////////////////////////
// FACTORY METHODS
////////////////////////////////////////
FeaturePtr Feature::Create(ftype x, ftype y) {
  auto f = MemoryManager::instance()->GetFeature();
  CHECK(f);
  f->Reset(x, y);
  return f;
}

void Feature::Delete(FeaturePtr f) {
  MemoryManager::instance()->ReturnFeature(f);
}

void Feature::Reset(ftype x, ftype y) {
  id_ = counter_++;
  sind_ = -1;
  init_counter_ = 0;
  lifetime_ = 0;
  status_ = FeatureStatus::CREATED;
  ref_ = nullptr;
  Track::Reset(x, y);
  x_ << x, y, 2.0;
  pred_ << -1, -1;
  J_.setZero();
  inn_ << 0, 0;
  outlier_counter_ = 0;

  sim_.Xs << -1, -1, -1;
  sim_.xp << -1, -1;
  sim_.xc << -1, -1;
  sim_.z = -1;
  sim_.lifetime = -1;
}

////////////////////////////////////////
// SOME ACCESSORS
////////////////////////////////////////
Vec3 Feature::Xc(Mat3 *J) const {
#ifdef USE_INVDEPTH
  return unproject_invz(x_, J);
#else
  return unproject_logz(x_, J);
#endif
}

Vec3 Feature::Xs(const SE3 &gbc, Mat3 *J) const {
  // Rsb * (Rbc*Xc + Tbc) + Tsb
  CHECK(ref_) << "feature #" << id_ << " null ref";
  SE3 gsc = ref_->gsb() * gbc;
  Vec3 Ps = gsc * Xc(J); // J = dXc_dx, where x is the local parametrization
  if (J) {
    *J = gsc.R().matrix() * (*J);
  }
  return Ps;
}

ftype Feature::z() const {
#ifdef USE_INVDEPTH
  // FIXME: ensure depth is positive
  return 1.0 / x_(2);
#else
  // in log-depth parametrization, positivity is guaranteed
  return exp(x_(2));
#endif
}

bool Feature::instate() const { return status_ == FeatureStatus::INSTATE; }

ftype Feature::score() const {
  CHECK(!instate())
      << "score function should only be called for feature not-instate yet";
  // TODO: come up with better scoring
  // confidence (negative uncertainty) in depth as score
  // return -P_(0, 0) * P_(1, 1) * P_(2, 2);
  return -P_(2, 2);
}

void Feature::Initialize(ftype z0, const Vec3 &std_xyz) {
  x_.head<2>() = Camera::instance()->UnProject(back());
#ifdef USE_INVDEPTH
  x_(2) = 1.0 / z0;
#else
  x_(2) = log(z0);
#endif

  // ftype rho = 1.0 / z0;
  // ftype rho_max = std::max(1.0 / (z0 - std_xyz(2)), 0.10);  // 0.10 is
  // inverse of max possible depth
  // ftype rho_min = 1.0 / (z0 + std_xyz(2));
  // ftype std_rho = std::max(fabs(rho - rho_min), fabs(rho - rho_max));

  P_ << std_xyz(0), 0, 0, 0, std_xyz(1), 0, 0, 0, std_xyz(2);
  P_ *= P_;
  status_ = FeatureStatus::INITIALIZING;
}

void Feature::SetRef(GroupPtr ref) {
  CHECK(ref_ == nullptr) << "reference already set!";
  // be very careful when reset references
  VLOG(0) << "ref group# " << ref->id() << " -> feature #" << id_;
  ref_ = ref;
}

void Feature::ResetRef(GroupPtr nref) {

  std::string str;
  if (nref == nullptr) {
    str = "nullptr";
  } else {
    str = "#" + std::to_string(nref->id());
  }

  VLOG(0) << "feature #" << id_ << " reset ref from #" << ref_->id() << " to "
          << str;

  ref_ = nref;
}

void Feature::SubfilterUpdate(const SE3 &gsb, const SE3 &gbc,
                              const SubfilterOptions &options) {

  CHECK(track_status() == TrackStatus::TRACKED);
  CHECK(status_ == FeatureStatus::INITIALIZING ||
        status_ == FeatureStatus::READY);

  init_counter_++;
  // depth sub-filter update
  Mat3 dXc_dx;
  Vec3 Xc = this->Xc(&dXc_dx);
  SE3 gtot = (gsb * gbc).inv() * ref()->gsb() * gbc; // g(curr cam <- ref cam)
  Vec3 Xcn = gtot * Xc;                              // predicted Xc
  Mat3 dXcn_dXc = gtot.rotation();
  Mat23 dxcn_dXcn;
  Vec2 xcn = project(Xcn, &dxcn_dXcn);

  Mat2 dxp_dxcn;
  Vec2 xp = Camera::instance()->Project(xcn, &dxp_dxcn);

  Mat23 H = dxp_dxcn * dxcn_dXcn * dXcn_dXc * dXc_dx;
  Vec2 inn = this->xp() - xp;

  Mat2 S = H * P_ * H.transpose();
  ftype Rtri = options.Rtri;
  S(0, 0) += Rtri;
  S(1, 1) += Rtri;

  ftype ratio{inn.dot(S.llt().solve(inn)) / options.MH_thresh};

  if (ratio > 1) {
    S(0, 0) += Rtri * (ratio - 1);
    S(1, 1) += Rtri * (ratio - 1);
    outlier_counter_ += sqrt(ratio);
  } else {
    outlier_counter_ = 0;
  }

  Mat32 K = P_ * H.transpose() * S.inverse(); // kalman gain

  x_ += K * inn;
  Mat3 I_KH = Mat3::Identity() - K * H;
  P_ = I_KH * P_ * I_KH.transpose() + K * Rtri * K.transpose();

  if (init_counter_ > options.ready_steps) {
    SetStatus(FeatureStatus::READY);
  } else {
    SetStatus(FeatureStatus::INITIALIZING);
  }
}

bool Feature::RefineDepth(const SE3 &gbc,
                          const std::vector<Observation> &observations,
                          const RefinementOptions &options) {

  std::vector<Observation> views;
  if (options.two_view) {
    auto[first, last] =
        std::minmax_element(std::begin(observations), std::end(observations),
                            [](const Observation &o1, const Observation &o2) {
                              return o1.g->id() < o1.g->id();
                            });
    views = {*first, *last};
  } else {
    views = observations;
  }

  Mat3 FtF;
  Vec3 Ftr;           // F.transpose() * residual
  ftype res_norm0{0}; // norm of residual corresponding to optimal state

  for (int iter = 0; iter < options.max_iters; ++iter) {
    Mat3 dXs_dx;
    Vec3 Xs = this->Xs(gbc, &dXs_dx); // ref_->gsb() * gbc * this->Xc();

    FtF.setZero();
    Ftr.setZero();
    ftype res_norm{0};

    for (const auto &obs : views) {
      // skip reference group
      if (obs.g->id() == ref_->id())
        continue;

      SE3 g_cn_s = (obs.g->gsb() * gbc).inv(); // spatial -> camera new
      Vec3 Xcn = g_cn_s * Xs;
      // Mat3 dXc_dXs = gcs.rotation();
      Mat3 dXcn_dx = g_cn_s.R().matrix() * dXs_dx;

      Mat23 dxcn_dXcn;
      Vec2 xcn = project(Xcn, &dxcn_dXcn);

      Mat2 dxp_dxcn;
      Vec2 xp = Camera::instance()->Project(xcn, &dxp_dxcn);

      Mat23 dxp_dx = dxp_dxcn * dxcn_dXcn * dXcn_dx;
      // std::cout << dxp_dx << std::endl;

      FtF += dxp_dx.transpose() * dxp_dx;
      Vec2 res = obs.xp - xp;
      Ftr += dxp_dx.transpose() * res;

      res_norm += res.norm();
    }

    if (iter > 0 && res_norm > res_norm0) {
      // current state not good, revert
      RestoreState();
      break;
    }

    VLOG_IF(0, iter > 0) << absl::StrFormat("iter=%d; |res|:%0.4f->%0.4f", iter,
                                            res_norm0, res_norm);

    Vec3 delta = FtF.ldlt().solve(Ftr);
    BackupState();
    x_ += delta;
    res_norm0 = res_norm;

    // not much to progress
    if (delta.lpNorm<Eigen::Infinity>() < options.eps) {
      break;
    }
  }

  if (auto res_norm_per_obs{res_norm0 / observations.size()};
      res_norm_per_obs > options.max_res_norm) {
    VLOG(0) << absl::StrFormat("feature #%d; status=%d; |res|=%f\n", id_,
                               as_integer(status_), res_norm_per_obs);
    return false;
  } else {
    return true;
  }
}

void Feature::ComputeJacobian(const Mat3 &Rsb, const Vec3 &Tsb, const Mat3 &Rbc,
                              const Vec3 &Tbc, const Vec3 &gyro, const Mat3 &Cg,
                              const Vec3 &bg, const Vec3 &Vsb, ftype td) {

  Mat3 Rsb_t = Rsb.transpose();
  Mat3 Rbc_t = Rbc.transpose();

  Mat3 Rr = ref_->Rsb();
  Vec3 Tr = ref_->Tsb();

  cache_.Xc = Xc(&cache_.dXc_dx);

  // Xc(ref) to Xs
  cache_.Xs = Rr * Rbc * cache_.Xc + Rr * Tbc + Tr;
  // cache_.dXs_dXc = Rr * Rbc;
  cache_.dXs_dx = Rr * Rbc * cache_.dXc_dx;
  cache_.dXs_dTbc = Rr;
  cache_.dXs_dWbc = -Rr * hat(cache_.Xc);
  cache_.dXs_dTr = Mat3::Identity();
  cache_.dXs_dWr = -hat(Rbc * cache_.Xc + Tbc);

  // Xs back to Xc(new)
  cache_.Xcn = Rbc_t * Rsb_t * (cache_.Xs - Tsb) - Rbc_t * Tbc;
  cache_.dXcn_dWbc = hat(Rsb_t * (cache_.Xs - Tsb) - Tbc);
  cache_.dXcn_dWsb = Rbc_t * hat(cache_.Xs - Tsb);
  cache_.dXcn_dXs = Rbc_t * Rsb_t; // dXcn_d... = dXcn_dXs * dXs_d...
  cache_.dXcn_dTsb = -cache_.dXcn_dXs;
  cache_.dXcn_dTbc = -Rbc_t;

#ifdef USE_ONLINE_TEMPORAL_CALIB
  Vec3 gyro_calib = Cg * gyro - bg;
  cache_.dXcn_dtd =
      -Rbc_t * (hat(gyro_calib) * Rsb_t * cache_.Xs + Rsb_t * Vsb);

  // since imu.Cg is used here, also need to compute jacobian block w.r.t. Cg
  auto dXcn_dW =
      dAB_dB(Rbc_t * hat(Rsb_t * cache_.Xs) * td, Vec3{}); // W=Cg * Wm
#ifdef USE_ONLINE_IMU_CALIB
  Eigen::Matrix<ftype, 3, 9> dW_dCg;
  for (int i = 0; i < 3; ++i) {
    dW_dCg.block<1, 3>(i, 3 * i) = gyro;
  }
  cache_.dXcn_dCg = dXcn_dW * dW_dCg;
#endif
  cache_.dXcn_dbg = dXcn_dW;
#endif

  // Rbc and Tbc are used twice, so add extra terms
  cache_.dXcn_dWbc += cache_.dXcn_dXs * cache_.dXs_dWbc;
  cache_.dXcn_dTbc += cache_.dXcn_dXs * cache_.dXs_dTbc;

  cache_.dXcn_dx = cache_.dXcn_dXs * cache_.dXs_dx;
  cache_.dXcn_dWr = cache_.dXcn_dXs * cache_.dXs_dWr;
  cache_.dXcn_dTr = cache_.dXcn_dXs * cache_.dXs_dTr;

  // xc(new)
  cache_.xcn = project(cache_.Xcn, &cache_.dxcn_dXcn);

#ifdef USE_ONLINE_CAMERA_CALIB
  Eigen::Matrix<ftype, 2, -1> jacc;
  cache_.xp = Camera::instance()->Project(cache_.xcn, &cache_.dxp_dxcn, &jacc);
#endif

  cache_.dxp_dXcn = cache_.dxp_dxcn * cache_.dxcn_dXcn;

  // set jacobians
  J_.setZero();
  J_.block<2, 3>(0, Index::Wsb) = cache_.dxp_dXcn * cache_.dXcn_dWsb;
  J_.block<2, 3>(0, Index::Tsb) = cache_.dxp_dXcn * cache_.dXcn_dTsb;
  J_.block<2, 3>(0, Index::Wbc) = cache_.dxp_dXcn * cache_.dXcn_dWbc;
  J_.block<2, 3>(0, Index::Tbc) = cache_.dxp_dXcn * cache_.dXcn_dTbc;
#ifdef USE_ONLINE_TEMPORAL_CALIB
  J_.block<2, 1>(0, Index::td) = cache_.dxp_dXcn * cache_.dXcn_dtd;
#ifdef USE_ONLINE_IMU_CALIB
  J_.block<2, 9>(0, Index::Cg) = cache_.dxp_dXcn * cache_.dXcn_dCg;
#endif
  J_.block<2, 3>(0, Index::bg) = cache_.dxp_dXcn * cache_.dXcn_dbg;
#endif
  CHECK(ref_->sind() != -1);
  CHECK(sind() != -1);
  int goff = kGroupBegin + 6 * ref_->sind();
  int foff = kFeatureBegin + 3 * sind();

  J_.block<2, 3>(0, goff) = cache_.dxp_dXcn * cache_.dXcn_dWr;
  J_.block<2, 3>(0, goff + 3) = cache_.dxp_dXcn * cache_.dXcn_dTr;
  J_.block<2, 3>(0, foff) = cache_.dxp_dXcn * cache_.dXcn_dx;

#ifdef USE_ONLINE_CAMERA_CALIB
  // fill-in jacobian w.r.t. camera intrinsics
  int dim{Camera::instance()->dim()};
  J_.block(0, kCameraBegin, 2, dim) = jacc.block(0, 0, 2, dim);
#endif

  // innovation
  cache_.inn = back() - cache_.xp;
  inn_ = cache_.inn;
}

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
  cache_.dXcn_dWsb = Rbc_t * hat(cache_.Xs - Tsb);
  cache_.dXcn_dTbc = -Rbc_t;
  cache_.dXcn_dWbc = hat(Rsb_t * (cache_.Xs - Tsb) - Tbc);

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

void Feature::FillJacobian(MatX &H, Vec2 &inn) {
  H = J_;
  inn = inn_;
}

void Feature::Triangulate(const SE3 &gsb, const SE3 &gbc,
                          const TriangulateOptions &options) {
  CHECK(size() == 2);
  Vec2 xc1 = CameraManager::instance()->UnProject(front());
  Vec2 xc2 = CameraManager::instance()->UnProject(back());
  SE3 g12 = (ref_->gsb() * gbc).inv() * (gsb * gbc);
  Vec3 Xc1 = options.method == 1 ? Triangulate1(g12, xc1, xc2)
                                 : Triangulate2(g12, xc1, xc2);

  if (auto z = Xc1(2); z < options.zmin || z > options.zmax) {
    // triangulated depth is not great
    // stick to the constant depth
  } else {
    x_.head<2>() = Xc1.head<2>() / z;
#ifdef USE_INVDEPTH
    x_(2) = 1.0 / z;
#else
    x_(2) = log(z);
#endif
  }
}

} // feh
