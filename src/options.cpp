#include "options.h"
#include "feature.h"

namespace feh {

bool Criteria::Candidate(FeaturePtr f) {
  ftype max_outlier_counter{0.01}; // FIXME (xfei): make a parameter
  return (f->status() == FeatureStatus::READY ||
          f->status() == FeatureStatus::INITIALIZING) &&
         (f->outlier_counter() < max_outlier_counter);
}

bool Criteria::CandidateStrict(FeaturePtr f) {
  ftype max_outlier_counter{0.01}; // FIXME (xfei): make a parameter
  return f->status() == FeatureStatus::READY &&
         (f->outlier_counter() < max_outlier_counter);
}

bool Criteria::CandidateComparison(FeaturePtr f1, FeaturePtr f2) {
  int s1 = as_integer(f1->status());
  int s2 = as_integer(f2->status());
  return (s1 > s2) || (s1 == s2 && f1->score() > f2->score());
}

} // namespace feh
