// Adapt feature and group objects to objects compatible with 
// the optimizer.
#pragma once
#include "optimizer_types.h"
#include "feature.h"
#include "group.h"

namespace xivo {

namespace adapter {

void AddFeature(FeaturePtr f);
void AddGroup(GroupPtr g);

} // namespace adapter

} // namespace xivo


