#pragma once

#include "feature.h"
#include "group.h"


namespace xivo {

class GraphBase {

public:

  void AddFeature(FeaturePtr f);
  void AddGroup(GroupPtr g);

  void RemoveFeature(const FeaturePtr f);
  void RemoveFeatures(const std::vector<FeaturePtr> &);
  void RemoveGroup(const GroupPtr g);
  void RemoveGroups(const std::vector<GroupPtr> &);

  bool HasGroup(GroupPtr g) const;
  bool HasGroup(int gid) const;
  bool HasFeature(FeaturePtr f) const;
  bool HasFeature(int fid) const;

  FeaturePtr GetFeature(int fid) const;
  std::vector<FeaturePtr> GetFeatures() const;
  GroupPtr GetGroup(int gid) const;
  std::vector<GroupPtr> GetGroups() const;

  std::vector<FeaturePtr> GetFeaturesOf(GroupPtr g) const;
  std::vector<GroupPtr> GetGroupsOf(FeaturePtr f) const;

  const FeatureAdj &GetFeatureAdj(FeaturePtr f) const;
  const GroupAdj &GetGroupAdj(GroupPtr g) const;

  std::vector<FeaturePtr>
  GetFeaturesIf(std::function<bool(FeaturePtr)> pred) const;
  std::vector<GroupPtr> GetGroupsIf(std::function<bool(GroupPtr)> pred) const;
  std::vector<Observation> GetObservationsOf(FeaturePtr f) const;


protected:
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

};


}