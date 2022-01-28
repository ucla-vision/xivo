#include "mapper.h"
#include "feature.h"
#include "group.h"

#ifdef USE_G2O
#include "optimizer_adapters.h"
#endif



namespace xivo {


std::unique_ptr<Mapper> Mapper::instance_{nullptr};

Mapper* Mapper::instance() {  return instance_.get(); }


Mapper::Mapper(const Json::Value &cfg) {

  use_loop_closure_ = cfg.get("detectLoopClosures", false).asBool();
  std::string vocab_file =
    cfg.get("vocabulary", "cfg/ukbench10K_FASTBRIEF32.yml.gz").asString();

  uplevel_word_search_ = cfg.get("uplevel_word_search", 0).asInt();
  nn_dist_thresh_ = cfg.get("nn_dist_thresh", 20.0).asDouble();
  voc_ = new FastBriefVocabulary(vocab_file);
}


MapperPtr Mapper::Create(const Json::Value &cfg) {
  if (instance_ == nullptr) {
    instance_ = std::unique_ptr<Mapper>(new Mapper(cfg));
  }
  return instance_.get();
}


Mapper::~Mapper() {}


void Mapper::AddFeature(FeaturePtr f, const FeatureAdj& f_obs) {

#ifdef USE_G2O
  adapter::AddFeature(f);
#endif

  int fid = f->id();
  CHECK(!features_.count(fid)) << "feature #" << fid << " already in mapper";

  // Merge observations of feature
  features_mtx.lock();
  // Case 1: feature not in map. Just add it.
  int matched_map_feat = f->LoopClosureMatch();
  if (matched_map_feat == -1) {
    features_[fid] = f;
    feature_adj_[fid] = f_obs;
  }
  // Case 2: feature has a loop closure. Need to merge observations
  else {
    for (auto p: f_obs) {
      feature_adj_[matched_map_feat].insert({p.first, p.second});
    }
  }
  features_mtx.unlock();

  // TODO (stsuei): Change the estimated location of the original feature.
  // This wasn't done in Corvis, so I'm not going to do it yet either.
  // I think a weighted sum based on covariance might be the right way to go..?

  // Add all feature descriptors to invese index
  FeaturePtr fptr = (matched_map_feat == -1) ? f : features_[matched_map_feat];
  std::vector<FastBrief::TDescriptor> all_descriptors = f->GetAllDBoWDesc();
  for (auto desc: all_descriptors) {
    DBoW2::WordId wid = voc_->transform(desc);
    UpdateInverseIndex(wid, fptr);
  }

  // If the feature has been merged with a previous feature, then we will
  // destroy it and its slot in the MemoryManager will become uninitialized.
  if (matched_map_feat > -1) {
    Feature::Destroy(f);
    //std::cout << "feature #" << fid << " merged with feature " <<
    //  matched_map_feat << std::endl;
    LOG(INFO) << "feature #" << fid << " merged with feature " <<
      matched_map_feat;
  }
  else {
    LOG(INFO) << "feature #" << fid << " added to mapper";
    //std::cout << "feature #" << fid << " added to mapper" << std::endl;
  }
}


void Mapper::AddGroup(GroupPtr g, const GroupAdj& g_features) {

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


std::vector<FeaturePtr> Mapper::GetFeaturesOf(GroupPtr g) const {
  std::vector<FeaturePtr> out;
  for (int fid : group_adj_.at(g->id())) {
    if (HasFeature(fid)) {
      out.push_back(features_.at(fid));
    }
  }
  return out;
}


std::vector<GroupPtr> Mapper::GetGroupsOf(FeaturePtr f) const {
  std::vector<GroupPtr> out;
  for (const auto &obs : feature_adj_.at(f->id())) {
    if (HasGroup(obs.first)) {
      out.push_back(groups_.at(obs.first));
    }
  }
  return out;
}


void Mapper::UpdateInverseIndex(const DBoW2::WordId &word_id, FeaturePtr f) {
  if (InvIndex_.count(word_id) > 0) {
    InvIndex_[word_id].insert(f);
  }
  else {
    InvIndex_.insert({word_id, {f}});
  }
}

std::unordered_set<FeaturePtr> Mapper::GetLoopClosureCandidates(
  const DBoW2::WordId& word_id)
{
  const DBoW2::NodeId &node_id(voc_->getParentNode(word_id, uplevel_word_search_));
  std::vector<DBoW2::WordId> words_under_node;
  voc_->getWordsFromNode(node_id, words_under_node);

  std::unordered_set<FeaturePtr> ret;
  for (auto w: words_under_node) {
    std::unordered_set<FeaturePtr> features_in_word = InvIndex_[w];
    ret.merge(features_in_word);
  }
  return ret;
}


std::vector<LCMatch> Mapper::DetectLoopClosures(const std::vector<FeaturePtr>& instate_features)
{

  std::vector<LCMatch> matches;

  for (auto f: instate_features) {
    FastBrief::TDescriptor desc = f->GetDBoWDesc();

    // Convert descriptor to word and find other features that match to the
    // same word
    const DBoW2::WordId& word_id(voc_->transform(desc));
    std::unordered_set<FeaturePtr> other_matches =
      GetLoopClosureCandidates(word_id);

    // Only use the matches that close enough to the descriptor
    double distance = nn_dist_thresh_;
    FeaturePtr best_match = nullptr;
    for (auto f1: other_matches) {
      FastBrief::TDescriptor desc1 = f1->GetDBoWDesc();
      double d = FastBrief::distance(desc, desc1);
      if (d < distance) {
        distance = d;
        best_match = f1;
      }
    }

    if (best_match != nullptr) {
//      std::cout << "Mapper: matched feature " << f->id() << " to feature "
//        << best_match->id() << std::endl;
      LOG(INFO) << "Mapper: matched feature " << f->id() << " to feature "
        << best_match->id() << std::endl;
      matches.push_back(LCMatch(f, best_match));
      f->SetLCMatch(best_match->id());
    }
  }

  // If number of matches is at least 4, check with P3P RANSAC. Otherwise, don't
  // return anything
  if (matches.size() > 4) {
    //std::cout << "Mapper: matched " << matches.size() << " features" << std::endl;
    LOG(INFO) << "Mapper: matched " << matches.size() << " features" << std::endl;
  }
  else {
    matches.clear();
  }

  return matches;
}


}