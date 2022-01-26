// The Mapper module.
// Author: Stephanie Tsuei (stephanietsuei@ucla.edu)
#pragma once

#include <mutex>

#include "json/json.h"
#include "DBoW2/DBoW2.h"

#include "core.h"
#include "feature.h"
#include "group.h"
#include "graphbase.h"
#include "fastbrief.h"

namespace xivo {

/** Singleton class in charge of loop closures and pose-graph optimization.
 *  It manages relationships for all features and groups formerly in the
 *  filter. It also contains functions to search for loop closures and runs
 *  pose-graph optimization.
 */

using FastBriefVocabulary =
  DBoW2::TemplatedVocabulary<FastBrief::TDescriptor, FastBrief>;
using FastBriefDatabase =
  DBoW2::TemplatedDatabase<FastBrief::TDescriptor, FastBrief>;

class Mapper : public GraphBase {

public:
  ~Mapper();
  static MapperPtr Create(const Json::Value &cfg);
  static MapperPtr instance();

  void AddFeature(FeaturePtr f, const FeatureAdj& feature_adj);
  void AddGroup(GroupPtr f, const GroupAdj& group_adj);
  void RemoveFeature(const FeaturePtr f);
  void RemoveGroup(const GroupPtr g);

  std::vector<FeaturePtr> GetFeaturesOf(GroupPtr g) const;
  std::vector<GroupPtr> GetGroupsOf(FeaturePtr f) const;

  std::mutex features_mtx;
  std::mutex groups_mtx;

  bool UseLoopClosure() const { return use_loop_closure_; }
  void DetectLoopClosures(const std::vector<FastBrief::TDescriptor>& descriptors,
                          const std::vector<cv::KeyPoint>& kps);

private:
  Mapper() = default;
  Mapper(const Json::Value &cfg);
  Mapper(const Mapper &) = delete;
  Mapper &operator=(const Mapper &) = delete;

  static std::unique_ptr<Mapper> instance_;

  // Loop closure variables
  bool use_loop_closure_;
  FastBriefDatabase* db_;
};



}