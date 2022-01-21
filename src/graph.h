// The feature-group visibility graph.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <functional>
#include <list>
#include <unordered_map>

#include "core.h"
#include "feature.h"
#include "group.h"

namespace xivo {

/** Graph of all features and groups that are currently being tracked in the
 * EKF.
 */
class Graph {
public:
  static Graph* Create();
  static Graph* instance();

  void RemoveFeature(const FeaturePtr f);
  void RemoveFeatures(const std::vector<FeaturePtr> &);

  void RemoveGroup(const GroupPtr g);
  void RemoveGroups(const std::vector<GroupPtr> &);

  /** Add a new feature to the graph. Called when `f` is first detected by
   * the `Tracker`. */
  void AddFeature(FeaturePtr f);
  /** Add a new group to the graph. Called once every time a new RGB image
   * was acquired. */
  void AddGroup(GroupPtr g);

  /** Marks that feature `f` was visible at group `g`. Function is called
   * when a feature is first detected by the tracker, and whenever it continues
   * to be tracked by the `Tracker`. 
   */
  void AddGroupToFeature(GroupPtr g, FeaturePtr f);
  /** Marks that feature `f` was visible at group `g`. Function is called
   * when a feature is first detected by the tracker, and whenever it continues
   * to be tracked by the `Tracker`. 
   */
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
  // This function is called when a group `g` is discarded to make room for
  // more groups in the `MemoryManager`. Features that were first created at
  // group `g` have their coordinates changed to a new group.
  std::vector<FeaturePtr> TransferFeatureOwnership(GroupPtr g, const SE3 &gbc);

  // Helper function to `TransferFeatureOwnership`. Finds a new reference frame
  // for a feature when its group is removed from the graph to make room for new
  // groups.
  GroupPtr FindNewOwner(FeaturePtr f);

  /** Checks that
   * 1. Every member of `features_` has at least one observation
   * 2. The group in every `Observation` of all features is a member of `groups_`
   * 3. The reference group of every feature is not NULL
   * 4. The reference group of every feature is a member of `groups_`
   * 5. There is a set of features associated with each group
   * 6. Every feature associated with a group is a member of `features_`
   * (Function only used for debugging.)
  */
  void SanityCheck();

  /** Removes (and deletes) all groups that have no adjacent features. 
   * (Function not actually used.)
  */
  void CleanIsolatedGroups();
  /** Removes (and deletes all features) that have no observations in
   * `feature_adj_`.
   * (Function not actualy used.)
   */ 
  void CleanIsolatedFeatures();
  /** Calls `CleanIsolatedGroups()` and `CleanIsolatedFeatures()`.
   * (Function not actually used.)
   */
  void CleanIsolatedNodes();

private:
  Graph() = default;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  static std::unique_ptr<Graph> instance_;

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

} // namespace xivo
