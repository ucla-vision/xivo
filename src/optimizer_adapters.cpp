#include "optimizer_adapters.h"
#include "optimizer.h"
#include "graph.h"
#include "estimator.h"

namespace xivo {

namespace adapter {

void AddFeature(FeaturePtr f) {
  if (!Graph::instance()->HasFeature(f)) {
    return;
  }
  // adapt feature
  FeatureAdapter adapter_f{f->id(), f->Xs()};
  VectorObsAdapterG adapter_obs;

  // adapt observations
  auto gbc{Estimator::instance()->gbc()};
  std::vector<Obs> vobs = Graph::instance()->GetObservationsOf(f);
  for (const auto& obs : vobs) {
    GroupPtr g = obs.g;
    GroupAdapter adapter_g{g->id(), g->gsb() * gbc};
    Vec2 xc = Camera::instance()->UnProject(obs.xp);
    adapter_obs.push_back(std::make_tuple(adapter_g, xc, Mat2::Identity()));
  }
  std::cout << "adding feature #" << adapter_f.id << 
    " with " << adapter_obs.size() << " groups" << std::endl;
  Optimizer::instance()->AddFeature(adapter_f, adapter_obs);
}

void AddGroup(GroupPtr g) {
  if (!Graph::instance()->HasGroup(g)) {
    return;
  }

  // adapt group

  auto gbc{Estimator::instance()->gbc()};
  GroupAdapter adapter_g{g->id(), g->gsb() * gbc};
  VectorObsAdapterF adapter_obs;

  std::vector<FeaturePtr> vf = Graph::instance()->GetFeaturesOf(g);
  for (FeaturePtr f : vf) {
    Vec2 xp = Graph::instance()->GetFeatureAdj(f).at(g->id());
    Vec2 xc = Camera::instance()->UnProject(xp);
    FeatureAdapter adapter_f{f->id(), f->Xs()};
    adapter_obs.push_back(std::make_tuple(adapter_f, xc, Mat2::Identity()));
  }

  std::cout << "adding group #" << adapter_g.id << 
    " with " << adapter_obs.size() << " features" << std::endl;
  Optimizer::instance()->AddGroup(adapter_g, adapter_obs);
}

} // namespace adapter

} // namespace xivo
