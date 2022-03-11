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

  int num_mh_rejected = 0;
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
          num_mh_rejected++;
          f->SetStatus(FeatureStatus::REJECTED_BY_FILTER);
          LOG(INFO) << "feature #" << f->id() << " rejected by MH-gating";
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

  LOG(INFO) << "MH rejected " << num_mh_rejected << " features";

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


void Estimator::CloseLoop() {
  std::vector<FeaturePtr> instate_features =
    Graph::instance()->GetInstateFeatures();
  std::vector<LCMatch> matches;
  if (instate_features.size() > 0) {
    matches = Mapper::instance()->DetectLoopClosures(instate_features, gbc());
  }

  if (matches.size() > 0) {
    CloseLoopInternal(Graph::instance()->LastAddedGroup(), matches);
  }
}

void Estimator::CloseLoopInternal(GroupPtr g, std::vector<LCMatch>& matched_features) {

  Graph& graph{*Graph::instance()};

  int num_matches = matched_features.size();

  // H and R matrices
  int total_size = 2 * matched_features.size();
  H_.setZero(total_size, err_.size());
  diagR_.resize(total_size);
  inn_.setZero(total_size);

  // Compute feature Jacobians (fill in H)
  for (int i=0; i<num_matches; i++) {
    FeaturePtr new_feature = matched_features[i].first;
    FeaturePtr old_feature = matched_features[i].second;

    Observation obs = graph.GetObservationOf(new_feature, g);
    old_feature->ComputeLCJacobian(obs, X_.Rbc, X_.Tbc, err_, i, H_, inn_);

    // Fill in R
    diagR_.segment<2>(2*i) << Rlc_, Rlc_;

    // Print out stuffs
    //std::cout << "Comparing new (#" << new_feature->id() << ") to old (#" << old_feature->id() << ")" << std::endl;
    //std::cout << "new Xs: " << new_feature->Xs().transpose() << std::endl;
    //std::cout << "old Xs: " << old_feature->Xs().transpose() << std::endl;
  }

  //std::cout << "LC innovation: " << inn_.transpose() << std::endl;

  // Update Group list
  instate_groups_.clear();
  instate_groups_ = Graph::instance()->GetInstateGroups();

  // Measurement Update
  UpdateJosephForm();
  AbsorbError();
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

  /* We've already done the EKF prediction step and measurement prediction.
  So this step just looks for the maximal set of low-innovation inliers.
  */
  std::unordered_set<FeaturePtr> max_inliers, inliers;
  for (int i = 0; i < n_hyp && selected_counter < selected.size(); ++i) {
    int k = distribution(*rng_);
    while (selected[k]) {
      k = distribution(*rng_);
    }
    selected[k] = true;
    ++selected_counter;

    inliers.clear();
    for (auto f : mh_inliers) {
      auto res = f->xp() - f->Predict(gsb(), gbc());
      if (res.norm() < ransac_thresh_) {
        inliers.insert(f);
      }
    }
    if (inliers.size() > max_inliers.size()) {
      max_inliers = inliers;
      number_t eps = max_inliers.size() / float(mh_inliers.size());
      n_hyp = int(log(1 - ransac_prob_) / log(1-eps)) + 1; // RANSAC minimum number of trials
    }
  }
  auto str = StrFormat("#hyp tested=%d: li_inliers/mh_inliers=%d/%d",
                             n_hyp, max_inliers.size(), mh_inliers.size());

  LOG(INFO) << str;

  // If everything is a low-innovation inlier, we don't need to do anything more.
  if (max_inliers.size() == mh_inliers.size()) {
    return mh_inliers;
  }

  // Save which features are inliers.
  std::vector<bool> is_low_innovation_inlier;
  std::unordered_set<GroupPtr> groups_with_low_inn_inlier;
  for (int i=0; i<mh_inliers.size(); i++) {
    if (max_inliers.count(mh_inliers[i])) {
      is_low_innovation_inlier.push_back(true);
      groups_with_low_inn_inlier.insert(mh_inliers[i]->ref());
    }
    else {
      is_low_innovation_inlier.push_back(false);
    }
  }

  // back up state and covariance.
  BackupState(active_features, active_groups);


  // STEP 2: EKF update using only low-innovation inlier measurements.
  if (!max_inliers.empty()) {
    int size = err_.size();

    // Find a new temporary reference group if the gauge group pointer doesn't
    // contain a high-inlier feature
    if (groups_with_low_inn_inlier.count(gauge_group_ptr_) == 0) {
      LOG(INFO) << "One-Pt RANSAC using temporary new reference group";
      std::vector<GroupPtr> candidates;
      candidates.insert(candidates.end(), groups_with_low_inn_inlier.begin(),
                        groups_with_low_inn_inlier.end());
      GroupPtr tmpref = FindNewRefGroup(candidates);
      int offset = kGroupBegin + kGroupSize * tmpref->sind();
      P_.block(offset, 0, kGroupSize, size).setZero();
      P_.block(0, offset, size, kGroupSize).setZero();
    }

    // Zero out features and groups that aren't in the low-inlier set
    for (int i=0; i<mh_inliers.size(); i++) {
      if (!is_low_innovation_inlier[i]) {
        int offset = kFeatureBegin + kFeatureSize * mh_inliers[i]->sind();
        P_.block(offset, 0, kFeatureSize, size).setZero();
        P_.block(0, offset, size, kFeatureSize).setZero();
      }
    }
    for (auto g: active_groups) {
      if (groups_with_low_inn_inlier.count(g) == 0) {
        int offset = kGroupBegin + kGroupSize * g->sind();
        P_.block(offset, 0, kGroupSize, size).setZero();
        P_.block(0, offset, size, kGroupSize).setZero();
      }
    }

    // low innovation update
    H_.setZero(2 * max_inliers.size(), size);
    inn_.setZero(2 * max_inliers.size());
    diagR_.resize(2 * max_inliers.size());
    int f_cnt = 0;
    for (int i = 0; i < mh_inliers.size(); ++i) {
      if (is_low_innovation_inlier[i]) {
        H_.block(2 * f_cnt, 0, 2, size) = mh_inliers[i]->J();
        inn_.segment<2>(2 * f_cnt) = mh_inliers[i]->inn();
        diagR_.segment<2>(2 * f_cnt) << R_, R_;
        f_cnt++;
      }
    }
    UpdateJosephForm();
    AbsorbError();
  }

  if (max_inliers.size() < mh_inliers.size()) {
    // rescue high-innovation measurements
    std::vector<FeaturePtr> hi_inliers; // high-innovation inlier set
    for (int i = 0; i < mh_inliers.size(); ++i) {
      if (!is_low_innovation_inlier[i]) {
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
          LOG(INFO) << "feature #" << f->id() << " rejected by one-pt ransac";
        }
      }
    }
    if (!hi_inliers.empty()) {
      max_inliers.insert(hi_inliers.begin(), hi_inliers.end());
      LOG(INFO) << "rescued " << hi_inliers.size() << " high-innovation inliers"
                << std::endl;
    }

  }

  // restore state (need to re-compute jacobians at original state)
  RestoreState(active_features, active_groups);
  for (auto f : active_features) {
    f->ComputeJacobian(X_.Rsb, X_.Tsb, X_.Rbc, X_.Tbc, last_gyro_, imu_.Cg(),
                       X_.bg, X_.Vsb, X_.td, err_);
  }

  // create a vector for output
  std::vector<FeaturePtr> output;
  output.insert(output.end(), max_inliers.begin(), max_inliers.end());
  return output;
}

} // namespace xivo
