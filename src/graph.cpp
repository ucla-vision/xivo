#include "graph.h"
#include "estimator.h"
#include "feature.h"
#include "group.h"

#ifdef USE_G2O
#include "optimizer_adapters.h"
#endif

namespace xivo {

void FeatureAdj::Add(const Observation &obs) { insert({obs.g->id(), obs.xp}); }
void FeatureAdj::Remove(int id) { erase(id); }
void GroupAdj::Add(int id) { insert(id); }
void GroupAdj::Remove(int id) { erase(id); }

std::unique_ptr<Graph> Graph::instance_ = nullptr;

Graph* Graph::Create() {
  if (instance_ == nullptr) {
    instance_ = std::unique_ptr<Graph>(new Graph);
  } 
  return instance_.get();
}

Graph* Graph::instance() {
  if (instance_ == nullptr) {
    LOG(WARNING) << "Graph not created yet! Creating one ...";
    Graph::Create();
  } 
  return instance_.get();
}

bool Graph::HasGroup(GroupPtr g) const { return HasGroup(g->id()); }

bool Graph::HasGroup(int gid) const {
  return groups_.count(gid) && group_adj_.count(gid);
}

bool Graph::HasFeature(FeaturePtr f) const { return HasFeature(f->id()); }

bool Graph::HasFeature(int fid) const {
  return features_.count(fid) && feature_adj_.count(fid);
}

FeaturePtr Graph::GetFeature(int fid) const { return features_.at(fid); }

GroupPtr Graph::GetGroup(int gid) const { return groups_.at(gid); }

const FeatureAdj &Graph::GetFeatureAdj(FeaturePtr f) const {
  return feature_adj_.at(f->id());
}

const GroupAdj &Graph::GetGroupAdj(GroupPtr g) const {
  return group_adj_.at(g->id());
}

void Graph::RemoveFeature(const FeaturePtr f) {
  CHECK(HasFeature(f)) << "feature #" << f->id() << " not exists";

#ifdef USE_G2O
  adapter::AddFeature(f);
#endif

  int fid = f->id();
  features_.erase(fid);
  for (const auto &obs : feature_adj_.at(fid)) {
    group_adj_.at(obs.first).Remove(fid);
  }
  feature_adj_.erase(fid);

  LOG(INFO) << "feature #" << fid << " removed";
}

void Graph::RemoveFeatures(const std::vector<FeaturePtr> &features) {
  for (auto f : features) {
    RemoveFeature(f);
  }
}

void Graph::RemoveGroup(const GroupPtr g) {
  CHECK(HasGroup(g)) << "group #" << g->id() << " not exists";

#ifdef USE_G2O
  adapter::AddGroup(g);
#endif

  int gid = g->id();
  groups_.erase(gid);
  for (auto fid : group_adj_.at(gid)) {
    auto f = features_.at(fid);
    // need to transfer ownership of the feature first
    if (f->ref() == g) {
      LOG(FATAL) << "removing group #" << gid << " but feature #" << fid
                 << " refers to it";
    }
    feature_adj_.at(fid).Remove(gid);
  }
  group_adj_.erase(gid);
  LOG(INFO) << "group #" << gid << " removed";
}

void Graph::RemoveGroups(const std::vector<GroupPtr> &groups) {
  for (auto g : groups) {
    RemoveGroup(g);
  }
}

void Graph::AddFeature(FeaturePtr f) {
  int fid = f->id();
  CHECK(!features_.count(fid)) << "feature #" << fid << " arealdy exists";
  features_[fid] = f;
  feature_adj_[fid] = {};
  LOG(INFO) << "feature #" << fid << " added to graph";
}

void Graph::AddGroup(GroupPtr g) {
  int gid = g->id();
  CHECK(!groups_.count(gid)) << "group #" << gid << " already exists";
  groups_[gid] = g;
  group_adj_[gid] = {};
  LOG(INFO) << "group #" << gid << " added to graph";
}

void Graph::AddGroupToFeature(GroupPtr g, FeaturePtr f) {

  int gid = g->id();
  int fid = f->id();

  CHECK(HasFeature(f)) << "feature #" << fid << " not exists";
  CHECK(HasGroup(g)) << "group #" << gid << " not exists";

  feature_adj_.at(fid).Add({g, f->xp()});
  LOG(INFO) << "group #" << gid << " added to feature #" << fid;
}

void Graph::AddFeatureToGroup(FeaturePtr f, GroupPtr g) {

  int fid = f->id();
  int gid = g->id();

  CHECK(HasFeature(f)) << "feature #" << fid << " not exists";
  CHECK(HasGroup(g)) << "group #" << gid << " not exists";

  group_adj_[gid].Add(fid);
  LOG(INFO) << "feature #" << fid << " added to group #" << gid;
}

std::vector<FeaturePtr>
Graph::GetFeaturesIf(std::function<bool(FeaturePtr)> pred) const {
  std::vector<FeaturePtr> out;
  for (auto p : features_) {
    if (pred(p.second)) {
      out.push_back(p.second);
    }
  }
  return out;
}

std::vector<FeaturePtr> Graph::GetFeatures() const {
  std::vector<FeaturePtr> out;
  out.reserve(features_.size());
  for (auto p : features_) {
    out.push_back(p.second);
  }
  return out;
}

std::vector<GroupPtr> Graph::GetGroups() const {
  std::vector<GroupPtr> out;
  out.reserve(groups_.size());
  for (auto p : groups_) {
    out.push_back(p.second);
  }
  return out;
}

std::vector<FeaturePtr> Graph::GetFeaturesOf(GroupPtr g) const {
  std::vector<FeaturePtr> out;
  for (int fid : group_adj_.at(g->id())) {
    out.push_back(features_.at(fid));
  }
  return out;
}

std::vector<GroupPtr>
Graph::GetGroupsIf(std::function<bool(GroupPtr)> pred) const {
  std::vector<GroupPtr> out;
  for (auto p : groups_) {
    if (pred(p.second)) {
      out.push_back(p.second);
    }
  }
  return out;
}

std::vector<GroupPtr> Graph::GetGroupsOf(FeaturePtr f) const {
  std::vector<GroupPtr> out;
  for (const auto &obs : feature_adj_.at(f->id())) {
    out.push_back(groups_.at(obs.first));
  }
  return out;
}

std::vector<Observation> Graph::GetObservationsOf(FeaturePtr f) const {
  std::vector<Observation> out;
  for (const auto &obs : feature_adj_.at(f->id())) {
    out.push_back({groups_.at(obs.first), obs.second});
  }
  return out;
}

void Graph::SanityCheck() {
  for (auto p : features_) {
    int fid = p.first;
    auto f = p.second;
    CHECK(feature_adj_.count(fid));
    for (auto obs : feature_adj_.at(fid)) {
      CHECK(groups_.count(obs.first));
    }
    CHECK(f->ref());
    CHECK(groups_.count(f->ref()->id()));
  }

  for (auto p : groups_) {
    int gid = p.first;
    auto g = p.second;
    CHECK(group_adj_.count(gid));
    for (auto fid : group_adj_.at(gid)) {
      CHECK(features_.count(fid));
    }
  }
  LOG(INFO) << "#graph.features=" << features_.size()
            << " ;#graph.groups=" << groups_.size();
}

std::vector<FeaturePtr> Graph::TransferFeatureOwnership(GroupPtr g,
                                                        const SE3 &gbc) {

  CHECK(HasGroup(g));

  std::vector<FeaturePtr> failed;
  int gid = g->id();

  for (int fid : group_adj_.at(gid)) {
    CHECK(HasFeature(fid));

    auto f = features_.at(fid);
    if (f->ref() == g) {
      // transfer ownership
      auto nref = FindNewOwner(f);
      if (nref) {
        // now transfer
        SE3 g_cn_s =
            (nref->gsb() * gbc)
                .inv(); // spatial (s) to camera of the new reference (cn)
        Mat3 dXs_dx;
        Vec3 Xcn = g_cn_s * f->Xs(gbc, &dXs_dx);
        // Mat3 dXcn_dXs = gcb.R() * gbs.R();
        Mat3 dXcn_dx = g_cn_s.R().matrix() * dXs_dx;
        Mat3 dxn_dXcn;
#ifdef USE_INVDEPTH
        Vec3 xn = project_invz(Xcn, &dxn_dXcn);
#else
        if (Xcn(2) < 0) {
          f->ResetRef(nullptr);
          failed.push_back(f);
          LOG(WARNING) << "negative depth; mark feature #" << fid
                       << " as failed";
          continue;
        }
        Vec3 xn = project_logz(Xcn, &dxn_dXcn);
#endif
        f->x() = xn;

        Mat3 J;
        J = dxn_dXcn * dXcn_dx;

        f->P() =
            J * f->P() * J.transpose() * 1.5; // inflate covariance a little bit

        f->ResetRef(nref);

        LOG(INFO) << "feature #" << fid << " transfered from group #" << gid
                  << " to group #" << nref->id();
      } else {
        f->ResetRef(nullptr);
        failed.push_back(f);
        LOG(WARNING) << "failed to find new owner for feature #" << fid;
      }
    }
  }
  return failed;
}

GroupPtr Graph::FindNewOwner(FeaturePtr f) {
  int fid = f->id();
  CHECK(features_.count(fid));
  CHECK(feature_adj_.count(fid));
  auto old_gid = f->ref()->id();
  for (const auto &obs : feature_adj_.at(fid)) {
    int gid = obs.first;
    if (gid != old_gid) {
      // TODO: can have fancy measure on which group should be the best to be
      // the new owner
      // For now, just pick the first one met.
      return groups_.at(gid);
    }
  }
  return nullptr;
}

void Graph::CleanIsolatedGroups() {
  std::vector<GroupPtr> islands;
  ;
  for (auto p : groups_) {
    if (group_adj_.at(p.first).empty()) {
      islands.push_back(p.second);
    }
  }
  LOG(INFO) << "removing " << islands.size() << " isolated groups" << std::endl;
  RemoveGroups(islands);
  for (auto g : islands) {
    Group::Delete(g);
  }
}

void Graph::CleanIsolatedFeatures() {
  std::vector<FeaturePtr> islands;
  ;
  for (auto p : features_) {
    if (feature_adj_.at(p.first).empty()) {
      islands.push_back(p.second);
    }
  }
  LOG(INFO) << "removing " << islands.size() << " isolated features"
            << std::endl;
  RemoveFeatures(islands);
  for (auto f : islands) {
    Feature::Delete(f);
  }
}

void Graph::CleanIsolatedNodes() {
  CleanIsolatedFeatures();
  CleanIsolatedGroups();
}

} // namespace xivo
