#include "mapper.h"
#include "feature.h"
#include "group.h"

#ifdef USE_G2O
#include "optimizer_adapters.h"
#endif



namespace xivo {

std::unique_ptr<Mapper> Mapper::instance_{nullptr};

Mapper* Mapper::instance() {
  if (instance_ == nullptr) {
    LOG(WARNING) << "Mapper not created yet! Creating one...";
    Mapper::Create();
  }
  return instance_.get();
}


MapperPtr Mapper::Create() {
  if (instance_ == nullptr) {
    instance_ = std::unique_ptr<Mapper>(new Mapper);
  }
  return instance_.get();
}


Mapper::~Mapper() {}


void Mapper::AddFeature(FeaturePtr f, FeatureAdj& f_obs) {

#ifdef USE_G2O
  adapter::AddFeature(f);
#endif

  int fid = f->id();
  CHECK(!features_.count(fid)) << "feature #" << fid << " already in mapper";

  features_mtx.lock();
  features_[fid] = f;
  feature_adj_[fid] = f_obs;
  features_mtx.unlock();

  LOG(INFO) << "feature #" << fid << " added to mapper";
}


void Mapper::AddGroup(GroupPtr g, GroupAdj& g_features) {

#ifdef USE_G2O
  adapter::AddGroup(g);
#endif

  int gid = g->id();
  CHECK(!groups_.count(gid)) << "group #" << gid << " already in mapper";

  groups_mtx.lock();
  groups_[gid] = g;
  group_adj_[gid] = g_features;
  groups_mtx.unlock();

  LOG(INFO) << "group #" << gid << " added to mapper-graph";
}


void Mapper::RemoveFeature(const FeaturePtr f) {
  CHECK(HasFeature(f)) << "feature #" << f->id() << " not in mapper-graph";

  int fid = f->id();

  features_mtx.lock();
  features_.erase(fid);
  for (const auto &obs : feature_adj_.at(fid)) {
    if (HasGroup(obs.first)) {
      group_adj_.at(obs.first).Remove(fid);
    }
  }
  feature_adj_.erase(fid);
  features_mtx.unlock();

  LOG(INFO) << "feature #" << fid << " removed from mapper-graph";
}

void Mapper::RemoveGroup(const GroupPtr g) {
  CHECK(HasGroup(g)) << "group #" << g->id() << " not in mapper-graph";

  int gid = g->id();

  groups_mtx.lock();
  groups_.erase(gid);
  for (auto fid : group_adj_.at(gid)) {
    if (HasFeature(fid)) {
      auto f = features_.at(fid);

      // features should always get removed before removing the group
      if (f->ref() == g) {
        LOG(FATAL) << "mapper removing group #" << gid << " but feature #" << fid
                  << " refers to it";
      }

      feature_adj_.at(fid).Remove(gid);
    }
  }
  group_adj_.erase(gid);
  groups_mtx.unlock();

  LOG(INFO) << "group #" << gid << " removed from mapper-graph";
}


}