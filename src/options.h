// options objects for various depth-related algorithms
#pragma once
#include "core.h"

namespace feh {
// depth refinement options
struct RefinementOptions {
  RefinementOptions()
      : two_view{false}, max_iters{5}, eps{1e-5}, damping{1e-3},
        max_res_norm{5.0} {}

  bool two_view;
  int max_iters;      // maximal iterations to perform
  ftype eps;          // epsilon tolerance to stop optimization
  ftype damping;      // optional damping factor
  ftype max_res_norm; // maximal per observation residual norm
};

// depth subfilter options
struct SubfilterOptions {
  SubfilterOptions() : Rtri{3.5}, MH_thresh{5.991}, ready_steps{5} {}

  ftype Rtri;      // measurement covariance for triangulation
  ftype MH_thresh; // Mahalanobis gating threshold in depth-subfilter
  int ready_steps; // feature initialized with this amount of attempts is turned
                   // to ready status
};

// options for depth triangulation
struct TriangulateOptions {
  TriangulateOptions() : method{1}, zmin{0.05}, zmax{5.0} {}

  int method;
  ftype zmin, zmax;
};

struct Criteria {
  // how good is the feature to be an instate candidate
  static bool Candidate(FeaturePtr f);
  static bool CandidateStrict(FeaturePtr f);
  // return true if f1 is considered a better candidate than f2
  static bool CandidateComparison(FeaturePtr f1, FeaturePtr f2);
};

} // namespace feh
