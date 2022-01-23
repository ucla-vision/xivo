// The Mapper module.
// Author: Stephanie Tsuei (stephanietsuei@ucla.edu)
#pragma once

#include <mutex>

#include "core.h"
#include "feature.h"
#include "group.h"
#include "graphbase.h"

namespace xivo {

/** Singleton class in charge of loop closures and pose-graph optimization.
 *  It manages relationships for all features and groups formerly in the
 *  filter. It also contains functions to search for loop closures and runs
 *  pose-graph optimization.
 */
class Mapper : public GraphBase {

public:
  ~Mapper();
  static MapperPtr Create();
  static MapperPtr instance();

  void AddFeature(FeaturePtr f, FeatureAdj& feature_adj);
  void AddGroup(GroupPtr f, GroupAdj& group_adj);
  void RemoveFeature(const FeaturePtr f);
  void RemoveGroup(const GroupPtr g);

  std::vector<FeaturePtr> GetFeaturesOf(GroupPtr g) const;
  std::vector<GroupPtr> GetGroupsOf(FeaturePtr f) const;

  std::mutex features_mtx;
  std::mutex groups_mtx;

private:
  Mapper() = default;
  Mapper(const Mapper &) = delete;
  Mapper &operator=(const Mapper &) = delete;

  static std::unique_ptr<Mapper> instance_;

};



}