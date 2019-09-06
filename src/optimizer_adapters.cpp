#include "optimizer_adapters.h"
#include "optimizer.h"
#include "graph.h"

namespace xivo {

namespace adapter {

void AddFeature(FeaturePtr f) {
  if (!Graph::instance()->HasFeature(f)) {
    // std::cout << "feature #" << f->id() << " NOT found in graph" << std::endl;
    return;
  }
  // adapt feature
  FeatureAdapter adapter_f{f->id(), f->Xs()};
  VectorObsAdapterG adapter_obs;

  // adapt observations
  std::vector<Obs> vobs = Graph::instance()->GetObservationsOf(f);
  for (const auto& obs : vobs) {
    GroupPtr g = obs.g;
    GroupAdapter adapter_g{g->id(), g->gsb()};
    // FIXME (xfei): convert xp to bearing vector in body frame
    Vec2 xc = Camera::instance()->UnProject(obs.xp);
    // adapter_obs.push_back(std::make_tuple(adapter_g, xc, Mat2::Identity()));
    adapter_obs.push_back(ObsAdapterG{adapter_g, xc, Mat2::Identity()});
  }
  std::cout << "adding feature #" << adapter_f.id << 
    " with " << adapter_obs.size() << " groups" << std::endl;
  Optimizer::instance()->AddFeature(adapter_f, adapter_obs);
}

void AddGroup(GroupPtr g) {
  if (!Graph::instance()->HasGroup(g)) {
    // std::cout << "group #" << g->id() << " NOT found in graph" << std::endl;
    return;
  }

  // adapt group
  GroupAdapter adapter_g{g->id(), g->gsb()};
  VectorObsAdapterF adapter_obs;

  std::vector<FeaturePtr> vf = Graph::instance()->GetFeaturesOf(g);
  for (FeaturePtr f : vf) {
    Vec2 xp = Graph::instance()->GetFeatureAdj(f).at(g->id());
    Vec2 xc = Camera::instance()->UnProject(xp);
    // FIXME (xfei): convert xp to bearing vector in body frame
    FeatureAdapter adapter_f{f->id(), f->Xs()};
    adapter_obs.push_back(std::make_tuple(adapter_f, xc, Mat2::Identity()));
  }
  std::cout << "adding group #" << adapter_g.id << 
    " with " << adapter_obs.size() << " features" << std::endl;
  Optimizer::instance()->AddGroup(adapter_g, adapter_obs);
}

} // namespace adapter

} // namespace xivo
