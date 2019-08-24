#include "mm.h"
#include "feature.h"
#include "feature.h"
#include "group.h"

#include "absl/strings/str_format.h"
#include "glog/logging.h"

namespace xivo {

std::unique_ptr<MemoryManager> MemoryManager::instance_ = nullptr;

MemoryManagerPtr MemoryManager::Create(int max_features, int max_groups) {
  if (!instance_) {
    instance_ = std::unique_ptr<MemoryManager>(
        new MemoryManager(max_features, max_groups));
    LOG(INFO) << absl::StrFormat(
        "MemoryManager instance created with %d features and %d groups",
        max_features, max_groups);
  } else {
    LOG(WARNING) << "MemoryManager instance already created!";
  }
  return instance_.get();
}

MemoryManager::MemoryManager(int max_features, int max_groups) {
  for (int i = 0; i < max_features; ++i) {
    FeaturePtr addr = new Feature();
    fslots_[addr] = false;
  }
  for (int i = 0; i < max_groups; ++i) {
    GroupPtr addr = new Group();
    gslots_[addr] = false;
  }
}

MemoryManager::~MemoryManager() {
  for (auto p : fslots_) {
    delete p.first;
  }
  for (auto p : gslots_) {
    delete p.first;
  }
  // for (auto j : oos_jacs_) {
  //   delete j;
  // }
}

MemoryManagerPtr MemoryManager::instance() {
  if (!instance_) {
    throw std::runtime_error("MemoryManager instance not created yet");
  }
  return instance_.get();
}

FeaturePtr MemoryManager::GetFeature() {
  for (auto &p : fslots_) {
    if (!p.second) {
      p.second = true;
      return p.first;
    }
  }
  return nullptr;
}

void MemoryManager::ReturnFeature(FeaturePtr f) { fslots_.at(f) = false; }

GroupPtr MemoryManager::GetGroup() {
  for (auto &p : gslots_) {
    if (!p.second) {
      p.second = true;
      return p.first;
    }
  }
  return nullptr;
}

void MemoryManager::ReturnGroup(GroupPtr g) { gslots_.at(g) = false; }

} // namespace xivo
