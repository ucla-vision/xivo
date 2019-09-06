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
  auto adapter_f = FeatureAdapter{f->id(), f->Xs()};
  // adapt observations
  auto vobs = Graph::instance()->GetObservationsOf(f);
  std::vector<ObsAdapterG> adapter_obs;

  // for (const auto& obs : vobs) {
  //   auto g = obs.g;
  //   auto adapter_g = GroupAdapter{g->id(), g->gsb()};
  //   // FIXME (xfei): convert xp to bearing vector in body frame
  //   auto xc = Camera::instance()->UnProject(obs.xp);
  //   adapter_obs.push_back(std::make_tuple(adapter_g, xc, Mat2::Identity()));
  // }
  // std::cout << "adding feature #" << adapter_f.id << 
  //   " with " << adapter_obs.size() << " groups" << std::endl;
  Optimizer::instance()->AddFeature(adapter_f, adapter_obs);
}

void AddGroup(GroupPtr g) {
  if (!Graph::instance()->HasGroup(g)) {
    // std::cout << "group #" << g->id() << " NOT found in graph" << std::endl;
    return;
  }

  // adapt group
  auto adapter_g = GroupAdapter{g->id(), g->gsb()};

  std::vector<ObsAdapterF> adapter_obs;
  // auto vf = Graph::instance()->GetFeaturesOf(g);
  // for (auto f : vf) {
  //   auto xp = Graph::instance()->GetFeatureAdj(f).at(g->id());
  //   auto xc = Camera::instance()->UnProject(xp);
  //   // FIXME (xfei): convert xp to bearing vector in body frame
  //   FeatureAdapter adapter_f{f->id(), f->Xs()};
  //   adapter_obs.push_back(std::make_tuple(adapter_f, xc, Mat2::Identity()));
  // }
  // std::cout << "adding group #" << adapter_g.id << 
  //   " with " << adapter_obs.size() << " features" << std::endl;
  Optimizer::instance()->AddGroup(adapter_g, adapter_obs);
}

} // namespace adapter

} // namespace xivo
