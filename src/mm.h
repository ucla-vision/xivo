#pragma once
#include <memory>
#include <unordered_map>

#include "core.h"
#include "jac.h"

namespace xivo {

/** Singleton memory management for feature and groups.
 *  A fixed chunk of memory is pre-allocated for features and groups,
 *  which prevents memory leaks and frequent malloc calls.
 *  Author: Xiaohan Fei (feixh@cs.ucla.edu) */
class MemoryManager {
public:
  ~MemoryManager();

  static MemoryManagerPtr Create(int max_features, int max_groups);
  static MemoryManagerPtr instance();

  FeaturePtr GetFeature();
  void ReturnFeature(FeaturePtr);
  GroupPtr GetGroup();
  void ReturnGroup(GroupPtr);

private:
  MemoryManager() = delete;
  MemoryManager(const MemoryManager &) = delete;
  MemoryManager &operator=(const MemoryManager &) = delete;

  MemoryManager(int max_features = 512, int max_groups = 128);

  static std::unique_ptr<MemoryManager> instance_;
  std::unordered_map<FeaturePtr, bool> fslots_; // false = not used
  std::unordered_map<GroupPtr, bool> gslots_;
};

} // namespace xivo
