#include "mm.h"
#include "feature.h"
#include "group.h"
#include "mapper.h"

#include "glog/logging.h"

namespace xivo {

template<typename T>
CircBufWithHash<T>::CircBufWithHash(int max_items) {
  max_items_ = max_items;
  num_slots_initialized_ = 0;
  num_slots_active_ = 0;
  slot_search_ind_ = 0;

  for (int i=0; i < max_items_; i++) {
    T* addr = new T();
    slots_.push_back(addr);
    slots_initialized_.push_back(false);
    slots_active_.push_back(false);
    slots_map_[addr] = i;
  }
}

template<typename T>
CircBufWithHash<T>::~CircBufWithHash() {
  for (auto p: slots_map_) {
    delete p.first;
  }
}

template<typename T>
T* CircBufWithHash<T>::GetItem() {
  bool slot_found = false;

  // Try to use an slot that has never been used before first
  if (num_slots_initialized_ < max_items_) {
    while (!slot_found) {
      if (!slots_initialized_[slot_search_ind_]) {

#ifndef NDEBUG
        CHECK(!slots_active_[slot_search_ind_]);
#endif

        slots_initialized_[slot_search_ind_] = true;
        slots_active_[slot_search_ind_] = true;
        T* ret = slots_[slot_search_ind_];

        slot_found = true;
        num_slots_initialized_++;
        num_slots_active_++;

        slot_search_ind_ = (slot_search_ind_ + 1) % max_items_;

        return ret;
      }
      else {
        slot_search_ind_ = (slot_search_ind_ + 1) % max_items_;
      }
    }
  }

  // If all slots have been used before, just find one that isn't "active".
  else {
    while (!slot_found) {
      if (!slots_active_[slot_search_ind_]) {
        T* ret = slots_[slot_search_ind_];

        LOG(WARNING) << "MemoryManager: Overwriting inactive " << typeid(T).name()
          << " #" << ret->id();

        // Remove the item from the Mapper before doing anything else
        RemoveFromMapper(ret);

        slots_active_[slot_search_ind_] = true;

        slot_found = true;
        num_slots_active_++;

        slot_search_ind_ = (slot_search_ind_ + 1) % max_items_;
        return ret;
      }
      else {
        slot_search_ind_ = (slot_search_ind_ + 1) % max_items_;
      }
    }
  }

  return nullptr;
}


template<typename T>
void CircBufWithHash<T>::DeactivateItem(T *item) {
  int ind = slots_map_[item];
  slots_active_[ind] = false;
  num_slots_active_--;
}


template<typename T>
void CircBufWithHash<T>::DestroyItem(T *item) {
  int ind = slots_map_[item];
  slots_active_[ind] = false;
  slots_initialized_[ind] = false;
  num_slots_active_--;
  num_slots_initialized_--;
}


template<>
void CircBufWithHash<Feature>::RemoveFromMapper(FeaturePtr item) {
  Mapper::instance()->RemoveFeature(item);
}


template<>
void CircBufWithHash<Group>::RemoveFromMapper(GroupPtr item) {
  Mapper::instance()->RemoveGroup(item);
}


std::unique_ptr<MemoryManager> MemoryManager::instance_ = nullptr;

MemoryManagerPtr MemoryManager::Create(int max_features, int max_groups) {
  if (!instance_) {
    instance_ = std::unique_ptr<MemoryManager>(
        new MemoryManager(max_features, max_groups));
    LOG(INFO) << StrFormat(
        "MemoryManager instance created with %d features and %d groups",
        max_features, max_groups);
  } else {
    LOG(WARNING) << "MemoryManager instance already created!";
  }
  return instance_.get();
}

MemoryManager::MemoryManager(int max_features, int max_groups) {
  feature_slots_ = new CircBufWithHash<Feature>(max_features);
  group_slots_ = new CircBufWithHash<Group>(max_groups);
}

MemoryManager::~MemoryManager() {
  delete feature_slots_;
  delete group_slots_;
}

MemoryManagerPtr MemoryManager::instance() {
  if (!instance_) {
    throw std::runtime_error("MemoryManager instance not created yet");
  }
  return instance_.get();
}

FeaturePtr MemoryManager::GetFeature() {
  FeaturePtr addr = feature_slots_->GetItem();
  if (addr == nullptr) {
    LOG(FATAL) << "Memory Manager: Could not find an inactive feature slot."
      << std::endl;
  }
  return addr;
}

void MemoryManager::DeactivateFeature(FeaturePtr f) {
  feature_slots_->DeactivateItem(f);
}

void MemoryManager::DestroyFeature(FeaturePtr f) {
  feature_slots_->DestroyItem(f);
}

GroupPtr MemoryManager::GetGroup() {
  GroupPtr addr = group_slots_->GetItem();
  if (addr == nullptr) {
    LOG(FATAL) << "Memory manager: Could not find an inactive group slot."
      << std::endl;
  }
  return addr;
}

void MemoryManager::DeactivateGroup(GroupPtr g) {
  group_slots_->DeactivateItem(g);
}

void MemoryManager::DestroyGroup(GroupPtr g) {
  group_slots_->DestroyItem(g);
}

} // namespace xivo
