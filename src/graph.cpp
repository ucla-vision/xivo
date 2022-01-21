#include "graph.h"
#include "estimator.h"
#include "feature.h"
#include "group.h"
#include "mapper.h"


namespace xivo {

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


void Graph::RemoveFeature(const FeaturePtr f) {
  Mapper::instance()->AddFeature(f, feature_adj_.at(f->id()));
  GraphBase::RemoveFeature(f);
  LOG(INFO) << "feature #" << f->id() << " removed from Graph";
}

void Graph::RemoveFeatures(const std::vector<FeaturePtr> &features) {
  for (auto f : features) {
    RemoveFeature(f);
  }
}

void Graph::RemoveGroup(const GroupPtr g) {
  Mapper::instance()->AddGroup(g, group_adj_.at(g->id()));
  GraphBase::RemoveGroup(g);
  LOG(INFO) << "group #" << g->id() << " removed from Graph";
}

void Graph::RemoveGroups(const std::vector<GroupPtr> &groups) {
  for (auto g : groups) {
    RemoveGroup(g);
  }
}

void Graph::AddFeature(FeaturePtr f) {
  GraphBase::AddFeature(f);
  LOG(INFO) << "feature #" << f->id() << " added to graph";
}

void Graph::AddGroup(GroupPtr g) {
  GraphBase::AddGroup(g);
  LOG(INFO) << "group #" << g->id() << " added to graph";
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

        // TODO: Make covariance inflation factor a parameter.
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
