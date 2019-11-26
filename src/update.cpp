// The update step.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <unordered_set>

#include "glog/logging.h"

#ifdef USE_GPERFTOOLS
#include "gperftools/profiler.h"
#endif

#include "estimator.h"
#include "feature.h"
#include "geometry.h"
#include "group.h"
#include "tracker.h"

namespace xivo {

void Estimator::Update() {

#ifdef USE_GPERFTOOLS
  ProfilerStart(__PRETTY_FUNCTION__);
#endif

  if (instate_features_.empty() && oos_features_.empty())
    return;

  timer_.Tick("update");
  std::vector<FeaturePtr> inliers; // individually compatible matches
  std::vector<number_t> dist,
      inlier_dist; // MH distance of features & inlier features

  timer_.Tick("jacobian");
  for (auto f : instate_features_) {
    f->ComputeJacobian(X_.Rsb, X_.Tsb, X_.Rbc, X_.Tbc, last_gyro_, imu_.Cg(),
                       X_.bg, X_.Vsb, X_.td, err_);
    const auto &J = f->J();
    const auto &res = f->inn();

    // Mahalanobis gating
    Mat2 S = J * P_ * J.transpose();
    S(0, 0) += R_;
    S(1, 1) += R_;
    number_t mh_dist = res.dot(S.llt().solve(res));
    dist.push_back(mh_dist);
  }
  timer_.Tock("jacobian");

  timer_.Tick("MH-gating");

  if (use_MH_gating_ && instate_features_.size() > min_required_inliers_) {

    number_t mh_thresh = MH_thresh_;
    while (inliers.size() < min_required_inliers_) {
      // reset states
      for (auto f : instate_features_) {
        f->SetStatus(FeatureStatus::INSTATE);
      }
      inliers.clear();
      // mark inliers
      for (int i = 0; i < instate_features_.size(); ++i) {
        auto f = instate_features_[i];
        if (dist[i] < mh_thresh) {
          inliers.push_back(f);
        } else {
          f->SetStatus(FeatureStatus::REJECTED_BY_FILTER);
        }
      }
      // relax the threshold
      mh_thresh *= MH_thresh_multipler_;
    }
  } else {
    inliers.resize(instate_features_.size());
    std::copy(instate_features_.begin(), instate_features_.end(),
              inliers.begin());
  }
  timer_.Tock("MH-gating");

  if (use_1pt_RANSAC_) {
    inliers = OnePointRANSAC(inliers);
  }

  std::vector<FeaturePtr> active_oos_features;
  int total_oos_jac_size{0};
  if (use_OOS_) {
    // std::vector<OOSJacobian> oos_jacs; // jacobians w.r.t. feature
    // parametrization
    for (auto f : oos_features_) {
      auto vobs = Graph::instance()->GetObservationsOf(f);
      int oos_jac_size = f->ComputeOOSJacobian(vobs, X_.Rbc, X_.Tbc, err_);
      if (oos_jac_size > 0) {
        total_oos_jac_size += oos_jac_size;
        active_oos_features.push_back(f);
      }
    }
    if (total_oos_jac_size > 0) {
      LOG(INFO) << "#total_oos_jac=" << total_oos_jac_size << std::endl;
    }
  }

  if (inliers.empty() && (!use_OOS_ || !total_oos_jac_size)) {
    return;
  }

  int total_size = 2 * inliers.size() + total_oos_jac_size;
  H_.setZero(total_size, err_.size());
  inn_.setZero(total_size);
  diagR_.resize(total_size);

  for (int i = 0; i < inliers.size(); ++i) {
    inliers[i]->FillJacobianBlock(H_, 2 * i); 
    inn_.segment<2>(2 * i) = inliers[i]->inn();
    // if (outlier_thresh_ > 1.0) {
    //   auto [robust_R, is_outlier] = HuberOnInnovation(inliers[i]->inn(), R_);
    //   diagR_.segment<2>(2 * i) << robust_R, robust_R;
    // } else {
    //   diagR_.segment<2>(2 * i) << R_, R_;
    // }
    diagR_.segment<2>(2 * i) << R_, R_;
  }

  if (total_oos_jac_size) {
    int oos_offset = 2 * inliers.size();

    for (auto f : active_oos_features) {
      int size = f->oos_inn_size();
      H_.block(oos_offset, 0, size, err_.size()) = f->Ho();
      inn_.segment(oos_offset, size) = f->ro();
      for (int i = 0; i < size; ++i) {
        // FIXME (xfei): how to perform huber on innovation for OOS features?
        diagR_(oos_offset + i) = Roos_;
      }
      oos_offset += size;
    }
  }

  if (use_OOS_) {
    if (use_compression_ &&
        H_.rows() > H_.cols() * compression_trigger_ratio_) {
      // prform measurement compression
      int rows = QR(inn_, H_);
      inn_ = inn_.head(rows);
      H_ = H_.topRows(rows);
      diagR_ = diagR_.head(rows); // FIXME: this does not seem right
    }
  }

  timer_.Tick("actual-update");
  UpdateJosephForm();
  timer_.Tock("actual-update");

  // absorb error
  AbsorbError();
  timer_.Tock("update");

  LOG(INFO) << "Error state absorbed";

#ifdef USE_GPERFTOOLS
  ProfilerStop();
#endif
}

std::vector<FeaturePtr>
Estimator::OnePointRANSAC(const std::vector<FeaturePtr> &mh_inliers) {
  if (mh_inliers.empty())
    return mh_inliers;
  // Reference:
  // https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_jfr2010.pdf
  int n_hyp = 1000;
  std::vector<bool> selected(mh_inliers.size(), false);
  int selected_counter = 0;
  std::uniform_int_distribution<int> distribution(0, mh_inliers.size() - 1);

  // find those involved in update step
  std::unordered_set<FeaturePtr> active_features;
  std::unordered_set<GroupPtr> active_groups;
  for (auto f : mh_inliers) {
    active_features.insert(f);
  }
  for (auto f : mh_inliers) {
    active_groups.insert(f->ref());
  }

  // backup states
  State X0 = X_;
  for (auto g : active_groups) {
    g->BackupState();
  }
  for (auto f : active_features) {
    f->BackupState();
  }

  // tmp vars
  MatX K(err_.size(), 2);
  Mat2 S;

  std::vector<FeaturePtr> max_inliers, inliers;
  for (int i = 0; i < n_hyp && selected_counter < selected.size(); ++i) {
    int k = distribution(*rng_);
    while (selected[k]) {
      k = distribution(*rng_);
    }
    selected[k] = true;
    ++selected_counter;
    // state-only update with the selected measurement k
    //
    // std::cout << "H_.rows=" << H_.rows() << ";;; H_.cols=" << H_.cols() <<
    // std::endl;
    // std::cout << "index=" << mh_inliers[k].first << std::endl;

    auto J = mh_inliers[k]->J();
    auto inn = mh_inliers[k]->inn();

    S = J * P_ * J.transpose();
    S(0, 0) += R_;
    S(1, 1) += R_;

    K.transpose() = S.llt().solve(J * P_);

    err_ = K * inn;
    AbsorbError();
    err_.setZero(); // redudant (done in AbsorbError already) but for
                    // readibility

    inliers.clear();
    for (auto f : mh_inliers) {
      auto res = f->xp() - f->Predict(gsb(), gbc());

      // std::cout << "xp=" << f->xp().transpose() << std::endl;
      // std::cout << "pred=" << f->pred().transpose() << std::endl;
      // std::cout << "res=" << res.transpose() << std::endl;

      if (res.norm() < ransac_thresh_) {
        inliers.push_back(f);
      }
    }
    if (inliers.size() > max_inliers.size()) {
      max_inliers = inliers;
      number_t eps = 1 - max_inliers.size() / float(mh_inliers.size());
      n_hyp = int(log(1 - ransac_prob_) / log(eps + 1e-5)) + 1;
    }

    X_ = X0;
    for (auto f : active_features) {
      f->RestoreState();
    }
    for (auto g : active_groups) {
      g->RestoreState();
    }
  }
  auto str = StrFormat("#hyp tested=%d: li_inliers/mh_inliers=%d/%d",
                             n_hyp, max_inliers.size(), mh_inliers.size());

  LOG(INFO) << str;

  // back up state and covariance again
  MatX P0 = P_;
  for (auto g : active_groups) {
    g->BackupState();
  }
  for (auto f : active_features) {
    f->BackupState();
  }

  if (!max_inliers.empty()) {
    // low innovation update
    H_.resize(2 * max_inliers.size(), err_.size());
    H_.setZero();
    inn_.resize(2 * max_inliers.size());
    inn_.setZero();
    selected.resize(mh_inliers.size(), false);
    for (int i = 0; i < max_inliers.size(); ++i) {
      selected[i] = true;
      H_.block(2 * i, 0, 2, err_.size()) = max_inliers[i]->J();
      inn_.segment<2>(2 * i) = max_inliers[i]->inn();
    }
    UpdateJosephForm();
    AbsorbError();
  }

  if (max_inliers.size() < mh_inliers.size()) {
    // rescue high-innovation measurements
    std::vector<FeaturePtr> hi_inliers; // high-innovation inlier set
    for (int i = 0; i < mh_inliers.size(); ++i) {
      if (!selected[i]) {
        // potentially a high-innovation inlier
        auto f = mh_inliers[i];

        f->ComputeJacobian(X_.Rsb, X_.Tsb, X_.Rbc, X_.Tbc, last_gyro_,
                           imu_.Cg(), X_.bg, X_.Vsb, X_.td, err_);
        auto J = f->J();
        auto res = f->inn();

        Mat2 S = J * P_ * J.transpose();
        S(0, 0) += R_;
        S(1, 1) += R_;
        if (res.dot(S.llt().solve(res)) < ransac_Chi2_) {
          hi_inliers.push_back(f);
        } else {
          f->SetStatus(FeatureStatus::REJECTED_BY_FILTER);
        }
      }
    }
    if (!hi_inliers.empty()) {
      // H_.resize(2 * hi_inliers.size(), err_.size());
      // H_.setZero();
      // inn_.resize(2 * hi_inliers.size());
      // inn_.setZero();
      // for (int i = 0; i < hi_inliers.size(); ++i) {
      //   H_.block(i * 2, 0, 2, err_.size()) = hi_inliers[i]->J();
      //   inn_.segment<2>(i * 2) = hi_inliers[i]->inn();
      // }
      // UpdateJosephForm();
      // AbsorbError();
      //
      // // instead of using high-innovation inliers directly
      // // return them and perform one update
      max_inliers.insert(max_inliers.end(), hi_inliers.begin(),
                         hi_inliers.end());
      LOG(INFO) << "rescued " << hi_inliers.size() << " high-innovation inliers"
                << std::endl;
    }
  }

  // restore state
  X_ = X0;
  P_ = P0;
  for (auto f : active_features) {
    f->RestoreState();
  }
  for (auto g : active_groups) {
    g->RestoreState();
  }
  err_.setZero();
  // need to re-compute jacobians
  for (auto f : max_inliers) {
    f->ComputeJacobian(X_.Rsb, X_.Tsb, X_.Rbc, X_.Tbc, last_gyro_, imu_.Cg(),
                       X_.bg, X_.Vsb, X_.td, err_);
  }
  return max_inliers;
}

} // namespace xivo
