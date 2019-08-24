// Options objects for various depth-related algorithms,
// and policies for feature selection, etc.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "core.h"

namespace xivo {
// depth refinement options
struct RefinementOptions {
  RefinementOptions()
      : two_view{false}, max_iters{5}, eps{1e-5}, damping{1e-3},
        max_res_norm{5.0} {}

  bool two_view;
  int max_iters;      // maximal iterations to perform
  number_t eps;          // epsilon tolerance to stop optimization
  number_t damping;      // optional damping factor
  number_t max_res_norm; // maximal per observation residual norm
};

// depth subfilter options
struct SubfilterOptions {
  SubfilterOptions() : Rtri{3.5}, MH_thresh{5.991}, ready_steps{5} {}

  number_t Rtri;      // measurement covariance for triangulation
  number_t MH_thresh; // Mahalanobis gating threshold in depth-subfilter
  int ready_steps; // feature initialized with this amount of attempts is turned
                   // to ready status
};

// options for depth triangulation
struct TriangulateOptions {
  TriangulateOptions() : method{1}, zmin{0.05}, zmax{5.0} {}

  int method;
  number_t zmin, zmax;
};

struct Criteria {
  // how good is the feature to be an instate candidate
  static bool Candidate(FeaturePtr f);
  static bool CandidateStrict(FeaturePtr f);
  // return true if f1 is considered a better candidate than f2
  static bool CandidateComparison(FeaturePtr f1, FeaturePtr f2);
};

} // namespace xivo
