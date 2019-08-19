#include "group.h"
#include "mm.h"

namespace feh {

int Group::counter_ = 0;

////////////////////////////////////////
// FACTORY METHODS
////////////////////////////////////////
GroupPtr Group::Create(const SO3 &Rsb, const Vec3 &Tsb) {
  auto g = MemoryManager::instance()->GetGroup();
  CHECK(g);
  g->Reset(Rsb, Tsb);
  return g;
}

void Group::Delete(GroupPtr g) { MemoryManager::instance()->ReturnGroup(g); }

void Group::Reset(const SO3 &Rsb, const Vec3 &Tsb) {
  id_ = counter_++;
  lifetime_ = 0;
  sind_ = -1;
  status_ = GroupStatus::CREATED;
  X_.Rsb = Rsb;
  X_.Tsb = Tsb;
  VLOG(0) << "group #" << id_ << " created";
}

bool Group::instate() const {
  return status_ == GroupStatus::INSTATE || status_ == GroupStatus::GAUGE;
}

} // namespace feh
