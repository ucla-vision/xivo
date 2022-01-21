// The Mapper module.
// Author: Stephanie Tsuei (stephanietsuei@ucla.edu)
#pragma once

#include <mutex>

#include "core.h"
#include "feature.h"
#include "group.h"

namespace xivo {

/** Singleton class in charge of loop closures and pose-graph optimization.
 *  It manages relationships for all features and groups formerly in the
 *  filter. It also contains functions to search for loop closures and runs
 *  pose-graph optimization.
 */
class Mapper {

public:
  ~Mapper();
  static MapperPtr Create();
  static MapperPtr instance();

  void AddFeature(FeaturePtr f, FeatureAdj& feature_adj);
  void AddGroup(GroupPtr f, GroupAdj& group_adj);
  void RemoveFeature(const FeaturePtr f);
  void RemoveGroup(const GroupPtr g);

  std::mutex features_mtx;
  std::mutex groups_mtx;


private:
  Mapper() = default;
  Mapper(const Mapper &) = delete;
  Mapper &operator=(const Mapper &) = delete;

  static std::unique_ptr<Mapper> instance_;

  // 2 types of nodes: feature and group
  /** maps feature ids (int) to feature objects. */
  std::unordered_map<int, FeaturePtr> features_;
  /** maps group ids (int) to group objects. */
  std::unordered_map<int, GroupPtr> groups_;


  // adjaceny lists
  /** Unordered map: feature id (int) -> (group id -> observed pixel coords) */
  std::unordered_map<int, FeatureAdj> feature_adj_;

  /** Maps group id (int) to a unordered set of features visible when the
   * group was created. */
  std::unordered_map<int, GroupAdj> group_adj_;

  bool HasGroup(GroupPtr g) const;
  bool HasGroup(int gid) const;
  bool HasFeature(FeaturePtr f) const;
  bool HasFeature(int fid) const;


};



}