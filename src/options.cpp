// Options objects for various depth-related algorithms,
// and policies for feature selection, etc.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "options.h"
#include "feature.h"

namespace xivo {

bool Criteria::Candidate(FeaturePtr f) {
  number_t max_outlier_counter{0.01}; // FIXME (xfei): make a parameter
  bool good = (f->status() == FeatureStatus::READY ||
          f->status() == FeatureStatus::INITIALIZING) &&
         (f->outlier_counter() < max_outlier_counter);
  // FIXME: use zmin, zmax parameters
  good = good && (f->z() > 0.05 && f->z() < 5.0);
  return good;
}

bool Criteria::CandidateStrict(FeaturePtr f) {
  number_t max_outlier_counter{0.01}; // FIXME (xfei): make a parameter
  bool good = f->status() == FeatureStatus::READY &&
         (f->outlier_counter() < max_outlier_counter);
  // FIXME: use zmin, zmax parameters
  good = good && (f->z() > 0.05 && f->z() < 5.0);
  return good;
}

bool Criteria::CandidateComparison(FeaturePtr f1, FeaturePtr f2) {
  int s1 = as_integer(f1->status());
  int s2 = as_integer(f2->status());
  return (s1 > s2) || (s1 == s2 && f1->score() > f2->score());
}

} // namespace xivo
