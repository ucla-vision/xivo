// The feature-group visibility graph.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <functional>
#include <list>
#include <unordered_map>
#include <unordered_set>

#include "core.h"
#include "feature.h"

namespace xivo {

struct FeatureAdj : public std::unordered_map<int, Vec2> {
  void Add(const Observation &obs);
  void Remove(int id);
};

struct GroupAdj : public std::unordered_set<int> {
  void Add(int id);
  void Remove(int id);
};

class Graph {
public:
  static Graph* Create();
  static Graph* instance();

  void RemoveFeature(const FeaturePtr f);
  void RemoveFeatures(const std::vector<FeaturePtr> &);

  void RemoveGroup(const GroupPtr g);
  void RemoveGroups(const std::vector<GroupPtr> &);

  void AddFeature(FeaturePtr f);
  void AddGroup(GroupPtr g);

  void AddGroupToFeature(GroupPtr g, FeaturePtr f);
  void AddFeatureToGroup(FeaturePtr f, GroupPtr g);

  bool HasGroup(GroupPtr g) const;
  bool HasGroup(int gid) const;
  bool HasFeature(FeaturePtr f) const;
  bool HasFeature(int fid) const;
  FeaturePtr GetFeature(int fid) const;
  GroupPtr GetGroup(int gid) const;

  std::vector<FeaturePtr>
  GetFeaturesIf(std::function<bool(FeaturePtr)> pred) const;
  std::vector<FeaturePtr> GetFeatures() const;
  std::vector<GroupPtr> GetGroupsIf(std::function<bool(GroupPtr)> pred) const;
  std::vector<GroupPtr> GetGroups() const;

  std::vector<FeaturePtr> GetFeaturesOf(GroupPtr g) const;
  std::vector<GroupPtr> GetGroupsOf(FeaturePtr f) const;

  std::vector<Observation> GetObservationsOf(FeaturePtr f) const;

  const FeatureAdj &GetFeatureAdj(FeaturePtr f) const;
  const GroupAdj &GetGroupAdj(GroupPtr g) const;

  // transfer ownership of features owned by g, i.e., with g as reference
  // return those for which a new owner cannot be found
  // gsc is the camera to spatial frame transformation
  std::vector<FeaturePtr> TransferFeatureOwnership(GroupPtr g, const SE3 &gbc);
  GroupPtr FindNewOwner(FeaturePtr f);

  void SanityCheck();
  void CleanIsolatedGroups();
  void CleanIsolatedFeatures();
  void CleanIsolatedNodes();

private:
  Graph() = default;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  static std::unique_ptr<Graph> instance_;

  // 2 types of nodes: feature and group
  std::unordered_map<int, FeaturePtr> features_;
  std::unordered_map<int, GroupPtr> groups_;
  // adjacent list
  std::unordered_map<int, FeatureAdj> feature_adj_;
  std::unordered_map<int, GroupAdj> group_adj_;
};

} // namespace xivo
