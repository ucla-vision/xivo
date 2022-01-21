// The feature-group visibility graph.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <functional>
#include <list>
#include <unordered_map>

#include "core.h"
#include "feature.h"
#include "group.h"
#include "graphbase.h"

namespace xivo {

/** Graph of all features and groups that are currently being tracked in the
 * EKF.
 */
class Graph : public GraphBase {
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

};

} // namespace xivo
