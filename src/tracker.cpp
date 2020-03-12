// The feature tracking module;
// Multi-scale Lucas-Kanade tracker from OpenCV.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include <fstream>

#include "glog/logging.h"
#include "opencv2/video/video.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "feature.h"
#include "tracker.h"
#include "visualize.h"

namespace xivo {

std::unique_ptr<Tracker> Tracker::instance_ = nullptr;

TrackerPtr Tracker::Create(const Json::Value &cfg) {
  if (instance_ == nullptr) {
    instance_ = std::unique_ptr<Tracker>(new Tracker(cfg));
  } else {
    LOG(WARNING) << "tracker already created";
  }
  return instance_.get();
}

Tracker::Tracker(const Json::Value &cfg) : cfg_{cfg} {
  initialized_ = false;
  mask_size_ = cfg_.get("mask_size", 15).asInt();
  margin_ = cfg_.get("margin", 16).asInt();
  num_features_min_ = cfg_.get("num_features_min", 120).asInt();
  num_features_max_ = cfg_.get("num_features_max", 150).asInt();
  max_pixel_displacement_ = cfg_.get("max_pixel_displacement", 64).asInt();

  std::string optflow = cfg_.get("optical_flow_type", "LucasKanade").asString();
  if (optflow == "LucasKanade") {
    optflow_class_ = OpticalFlowType::LUCAS_KANADE;
    auto klt_cfg = cfg_["KLT"];
    lk_params_.win_size = klt_cfg.get("win_size", 15).asInt();
    lk_params_.max_level = klt_cfg.get("max_level", 4).asInt();
    lk_params_.max_iter = klt_cfg.get("max_iter", 15).asInt();
    lk_params_.eps = klt_cfg.get("eps", 0.01).asDouble();
  } 
  else if (optflow == "Farneback") {
    optflow_class_ = OpticalFlowType::FARNEBACK;
    auto fb_cfg = cfg_["FARNEBACK"];
    fb_params_.num_levels = fb_cfg.get("num_levels", 5).asInt();
    fb_params_.pyr_scale = fb_cfg.get("pyr_scale", 0.5).asDouble();
    fb_params_.win_size = fb_cfg.get("win_size", 13).asInt();
    fb_params_.num_iter = fb_cfg.get("num_iter", 10).asInt();
    fb_params_.polyN = fb_cfg.get("polyN", 5).asInt();

    std::string opencv_flag = fb_cfg.get("flags", "initial_flow").asString();
    if (opencv_flag == "initial_flow") {
      fb_params_.flags = cv::OPTFLOW_USE_INITIAL_FLOW;
    } else if (opencv_flag == "farneback_gaussian") {
      fb_params_.flags = cv::OPTFLOW_FARNEBACK_GAUSSIAN;
    } else {
      throw std::runtime_error("invalid Farneback flag. Expected [initial_flow|farneback_gaussian]");
    }

    farneback_flow_initialized_ = false;
  }
  else {
    throw NotImplemented();
  }


  std::string detector_type = cfg_.get("detector", "FAST").asString();
  LOG(INFO) << "detector type=" << detector_type;
  auto detector_cfg = cfg_[detector_type];

  if (detector_type == "FAST") {
    detector_ = cv::FastFeatureDetector::create(
        detector_cfg.get("threshold", 5).asInt(),
        detector_cfg.get("nonmaxSuppression", true).asBool());
  } else if (detector_type == "BRISK") {
    detector_ =
        cv::BRISK::create(detector_cfg.get("thresh", 5).asInt(),
                          detector_cfg.get("octaves", 3).asInt(),
                          detector_cfg.get("patternScale", 1.0).asFloat());
  } else if (detector_type == "ORB") {
    detector_ = cv::ORB::create(detector_cfg.get("nfeatures", 500).asInt(),
                                detector_cfg.get("scaleFactor", 1.2).asFloat(),
                                detector_cfg.get("nlevels", 4).asInt(),
                                detector_cfg.get("edgeThreshold", 31).asInt(),
                                detector_cfg.get("firstLevel", 0).asInt(),
                                detector_cfg.get("WTA_K", 2).asInt(),
                                detector_cfg.get("patchSize", 31).asInt(),
                                detector_cfg.get("fastThreshold", 20).asInt());
  } else if (detector_type == "AGAST") {
    detector_ = cv::AgastFeatureDetector::create(
        detector_cfg.get("threshold", 10).asInt(),
        detector_cfg.get("nonmaxSuppression", true).asBool());
  } else if (detector_type == "GFTT") {
    detector_ = cv::GFTTDetector::create(
        detector_cfg.get("maxCorners", 1000).asInt(),
        detector_cfg.get("qualityLevel", true).asDouble(),
        detector_cfg.get("minDistance", 1.0).asDouble(),
        detector_cfg.get("blockSize", 3).asInt(),
        detector_cfg.get("useHarrisDetector", false).asBool(),
        detector_cfg.get("k", 0.04).asDouble());
  } else {
    throw std::invalid_argument("unrecognized detector type");
  }
  LOG(INFO) << "detector created";

  descriptor_distance_thresh_ =
      cfg_.get("descriptor_distance_thresh", -1).asInt();
  extract_descriptor_ = cfg_.get("extract_descriptor", false).asBool() ||
                        descriptor_distance_thresh_ > -1;
  LOG(INFO) << "descriptor extraction " << extract_descriptor_ ? "ENABLED"
                                                               : "DISABLED";

  if (extract_descriptor_) {
    if (detector_type == "FAST" || detector_type == "AGAST" ||
        detector_type == "GFTT") {
      LOG(WARNING)
          << "detectors NOT able to extract descriptors; default to BRIEF";

      auto default_descriptor =
          cfg_.get("default_descriptor", "BRIEF").asString();
      auto desc_cfg = cfg_[default_descriptor];

      if (default_descriptor == "BRIEF") {
        extractor_ = cv::xfeatures2d::BriefDescriptorExtractor::create(
            desc_cfg.get("bytes", 64).asInt(),
            desc_cfg.get("use_orientation", false).asBool());
      } else if (default_descriptor == "FREAK") {
        extractor_ = cv::xfeatures2d::FREAK::create(
            desc_cfg.get("orientationNormalized", true).asBool(),
            desc_cfg.get("scaleNormalized", true).asBool(),
            desc_cfg.get("patternScale", 22.0).asDouble(),
            desc_cfg.get("nOctaves", 4).asInt());
      } else {
        throw std::invalid_argument("unrecognized descriptor type");
      }

    } else {
      // detector is also the extractor
      extractor_ = detector_;
    }
  }

  // Rescuing dropped tracks
  match_dropped_tracks_ = cfg_.get("match_dropped_tracks", false).asBool();
  if (match_dropped_tracks_ && !extract_descriptor_) {
    throw std::invalid_argument("must extract descriptors in order to match dropped tracks");
  }
  if (match_dropped_tracks_) {
    matcher_ = cv::BFMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING, true);
  }
}



void Tracker::Detect(const cv::Mat &img, int num_to_add) {
  std::vector<cv::KeyPoint> kps;
  detector_->detect(img, kps, mask_);
  // sort
  std::sort(kps.begin(), kps.end(),
            [](const cv::KeyPoint &kp1, const cv::KeyPoint &kp2) {
              return kp1.response > kp2.response;
            });

  cv::Mat descriptors;
  if (extract_descriptor_) {
    descriptors.reserveBuffer(kps.size() * 256);
    extractor_->compute(img, kps, descriptors);
  }

  // now every keypoint is equipped with a descriptor


  // match keypoints to old features - indices of these vectors correspond to
  // new features
  std::vector<bool> matched;
  std::vector<int> matchIdx;
  for (int i=0;  i<kps.size(); i++) {
    matched.push_back(false);
    matchIdx.push_back(-1);
  }

  if (match_dropped_tracks_ && (newly_dropped_tracks_.size() > 0)) {

    // Get matrix of old descriptors
    cv::Mat newly_dropped_descriptors(newly_dropped_tracks_.size(), 64, CV_8UC1);
    int i = 0;
    for (auto f: newly_dropped_tracks_) {
      newly_dropped_descriptors.row(i) = f->descriptor();
      i++;
    }

    // k-nearest neighbor match
    // query = newly-dropped descriptors
    // train = just-found descriptors
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(newly_dropped_descriptors, descriptors, knn_matches, 1);

    for (int i=0; i<knn_matches.size(); i++) {
      if (knn_matches[i].size() > 0) {
        matched[knn_matches[i][0].trainIdx] = true;
        matchIdx[knn_matches[i][0].trainIdx] = knn_matches[i][0].queryIdx;
      }
    }

  }

  // collect keypoints
  for (int i = 0; i < kps.size(); ++i) {
    const cv::KeyPoint &kp = kps[i];
    if (MaskValid(mask_, kp.pt.x, kp.pt.y)) {

      if (match_dropped_tracks_ && matched[i]) {
        int idx = matchIdx[i];
        FeaturePtr f1 = newly_dropped_tracks_[idx];
        f1->SetDescriptor(descriptors.row(i));
        f1->UpdateTrack(kp.pt.x, kp.pt.y);
        f1->SetTrackStatus(TrackStatus::TRACKED);
        LOG(INFO) << "Rescued dropped feature #" << f1->id();
        MaskOut(mask_, kp.pt.x, kp.pt.y, mask_size_);
        --num_to_add;
        continue;
      }

      // Didn't match to a previously-dropped track, so create a new feature
      FeaturePtr f = Feature::Create(kp.pt.x, kp.pt.y);
      features_.push_back(f);

      if (extract_descriptor_) {
        f->SetDescriptor(descriptors.row(i));
      }
      f->SetKeypoint(kp);

      // mask out
      MaskOut(mask_, kp.pt.x, kp.pt.y, mask_size_);
      --num_to_add;
    }
    if (num_to_add <= 0 || kp.response < 5)
      break;
  }
}


void Tracker::Update(const cv::Mat &image) {
  switch (optflow_class_) {
    case OpticalFlowType::LUCAS_KANADE:
      UpdatePyrLK(image);
      break;
    case OpticalFlowType::FARNEBACK:
      UpdateFarneback(image);
      break;
    default:
      throw std::runtime_error("Invalid Tracker Type, we shouldn't get here");
  }
}


void Tracker::InitializeTracker(const cv::Mat &image) {
  rows_ = image.rows;
  cols_ = image.cols;
  mask_ = cv::Mat(rows_, cols_, CV_8UC1);
  mask_.setTo(0);

  // build image pyramid if using Lucas Kanade
  if (optflow_class_ == OpticalFlowType::LUCAS_KANADE) {
    cv::buildOpticalFlowPyramid(img_, pyramid_, 
      cv::Size(lk_params_.win_size, lk_params_.win_size),
      lk_params_.max_level);
  }
  else if (optflow_class_ == OpticalFlowType::FARNEBACK) {
    farneback_flow_ = new cv::Mat(rows_, cols_, CV_32FC2);
  }

  // setup the mask
  ResetMask(mask_(
      cv::Rect(margin_, margin_, cols_ - 2 * margin_, rows_ - 2 * margin_)));
  // detect an initial set of features
  Detect(img_, num_features_max_);
  initialized_ = true;
  // std::cout << "tracker initialized";

}


void Tracker::UpdateFarneback(const cv::Mat &image) {

  if (!initialized_) {
    InitializeTracker(image);
    cv::Mat grey_image;
    cv::cvtColor(image, grey_image, CV_BGR2GRAY);
    img_ = grey_image.clone();
    if (cfg_.get("normalize", false).asBool()) {
      cv::normalize(grey_image, img_, 0, 255, cv::NORM_MINMAX);
    }
    return;
  }

  // convert image to greyscale
  cv::Mat grey_image;
  cv::cvtColor(image, grey_image, CV_BGR2GRAY);

  // Reset mask
  ResetMask(mask_(
      cv::Rect(margin_, margin_, cols_ - 2 * margin_, rows_ - 2 * margin_)));

  // Compute optical flow
  if (farneback_flow_initialized_) {
    cv::calcOpticalFlowFarneback(img_, grey_image, *farneback_flow_,
      fb_params_.pyr_scale, fb_params_.num_levels, fb_params_.win_size,
      fb_params_.num_iter, fb_params_.polyN, fb_params_.polySigma,
      cv::OPTFLOW_FARNEBACK_GAUSSIAN);
    farneback_flow_initialized_ = true;
  }
  else {
    cv::calcOpticalFlowFarneback(img_, grey_image, *farneback_flow_,
      fb_params_.pyr_scale, fb_params_.num_levels, fb_params_.win_size,
      fb_params_.num_iter, fb_params_.polyN, fb_params_.polySigma,
      fb_params_.flags);
  }

  // reserve memory for keypoints and descriptors
  std::vector<cv::KeyPoint> kps;
  std::vector<uint8_t> status;
  cv::Mat descriptors;
  std::vector<FeaturePtr> vf{features_.begin(), features_.end()};
  //kps.reserve(vf.size());
  //status.reserve(vf.size());
  descriptors.reserveBuffer(vf.size() * 256);

  // Translate old locations to new locations, save keypoints that are not to
  // be dropped
  for (int i = 0; i < vf.size(); ++i) {
    auto f = vf[i];
    cv::KeyPoint kp =
        f->keypoint(); // preserve all the properties of the initial keypoint

    // calculate updated pixel location
    number_t last_pix_x = kp.pt.x;
    number_t last_pix_y = kp.pt.y;

    cv::Vec2d flow = (*farneback_flow_).at<cv::Vec2d>(last_pix_y, last_pix_x);
    number_t curr_pix_x = last_pix_x + flow.val[0];
    number_t curr_pix_y = last_pix_y + flow.val[1];

    // Only compute descriptor if we haven't yet fallen off the image
    if (MaskValid(mask_, curr_pix_x, curr_pix_y) &&
        (Vec2{last_pix_x, last_pix_y} - Vec2{curr_pix_x, curr_pix_y}).norm()
          < max_pixel_displacement_) {
      status.push_back(1);
      kp.pt.x = curr_pix_x;
      kp.pt.y = curr_pix_y;
      kp.class_id = i;
      kps.push_back(kp);
    }
    else {
      status.push_back(0);
    }
  }

  // Extract descriptors
  // mark features that don't meet the descriptor distance threshold to be
  // dropped
  std::vector<cv::KeyPoint> kps_copy = kps;
  if (extract_descriptor_ && kps.size() > 0) {
    extractor_->compute(image, kps, descriptors);

    if (kps.size() < kps_copy.size()) {
      MatchKeypointsAndStatus(kps_copy, kps, status);
    }

    for (int i = 0; i < kps.size(); ++i) {
      auto f = vf[kps[i].class_id];
      if (descriptor_distance_thresh_ != -1) {
        int dist =
            cv::norm(f->descriptor(), descriptors.row(i), cv::NORM_HAMMING);
        if (dist > descriptor_distance_thresh_) {
          status[kps[i].class_id] = 0; // enforce to be dropped
        }
      }
      // FIXME: so if the distance test fails, we probably do not want to update
      // the descriptor
      // set new descriptor
      f->SetDescriptor(descriptors.row(i));
    }
  }

  // Update tracks of features that are good
  int num_valid_features = 0;
  for (int i=0; i<kps.size(); ++i) {
    auto f = vf[kps[i].class_id];
    if (status[kps[i].class_id]) {
      f->SetTrackStatus(TrackStatus::TRACKED);
      f->UpdateTrack(kps[i].pt.x, kps[i].pt.y);
      MaskOut(mask_, kps[i].pt.x, kps[i].pt.y, mask_size_);
      ++num_valid_features;
    }
  }

  // Toss out features that are no longer in the image
  newly_dropped_tracks_.clear();
  for (int i=0; i<status.size(); i++) {
    auto f = vf[i];
    if (!status[i]) {
      newly_dropped_tracks_.push_back(f);
    }
  }

  // detect a new set of features
  // this can rescue dropped featuers by matching them to newly detected ones
  if (num_valid_features < num_features_min_) {
    Detect(img_, num_features_max_ - num_valid_features);
  }

  // Mark all features that are still in newly_dropped_tracks_ at this point
  // as dropped
  for (auto f: newly_dropped_tracks_) {
    f->SetTrackStatus(TrackStatus::DROPPED);
  }

  // Save the image
  img_ = grey_image.clone();
  if (cfg_.get("normalize", false).asBool()) {
    cv::normalize(grey_image, img_, 0, 255, cv::NORM_MINMAX);
  }

}



void Tracker::UpdatePyrLK(const cv::Mat &image) {
  img_ = image.clone();
  if (cfg_.get("normalize", false).asBool()) {
    cv::normalize(image, img_, 0, 255, cv::NORM_MINMAX);
  }

  if (!initialized_) {
    InitializeTracker(image);
    return;
  }
  // reset mask
  ResetMask(mask_(
      cv::Rect(margin_, margin_, cols_ - 2 * margin_, rows_ - 2 * margin_)));
  // std::cout << "reset mask\n";

  // build new pyramid
  std::vector<cv::Mat> pyramid;
  cv::buildOpticalFlowPyramid(img_, pyramid,
    cv::Size(lk_params_.win_size, lk_params_.win_size),
    lk_params_.max_level);

  // prepare for optical flow
  cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                            lk_params_.max_iter, lk_params_.eps);

  std::vector<cv::Point2f> pts0, pts1;
  std::vector<uint8_t> status;
  std::vector<float> err;

  pts0.reserve(features_.size());
  pts1.reserve(pts0.size());

  for (auto f : features_) {
    const Vec2 &pt(f->xp());
    pts0.emplace_back(pt[0], pt[1]);

    // fill in predicted locations
    auto pred = f->pred();
    if (pred(0) != -1 && pred(1) != -1) {
      pts1.emplace_back(pred(0), pred(1));
      f->ResetPred(); // reset
    } else {
      pts1.emplace_back(pt[0], pt[1]);
    }
  }

  if (pts0.size() == 0) {
    initialized_ = false;
    return;
  }
  // cv::calcOpticalFlowPyrLK(pyramid_, pyramid, pts0, pts1, status, err,
  //                          cv::Size(win_size_, win_size_), max_level_,
  //                          criteria,
  //                          cv::OPTFLOW_USE_INITIAL_FLOW |
  //                          cv::OPTFLOW_LK_GET_MIN_EIGENVALS,
  //                          1e-4);
  cv::calcOpticalFlowPyrLK(pyramid_, pyramid, pts0, pts1, status, err,
                           cv::Size(lk_params_.win_size, lk_params_.win_size),
                           lk_params_.max_level, criteria,
                           cv::OPTFLOW_USE_INITIAL_FLOW);

  std::vector<cv::KeyPoint> kps;
  cv::Mat descriptors;
  if (extract_descriptor_) {
    std::vector<FeaturePtr> vf{features_.begin(), features_.end()};
    kps.reserve(vf.size());
    descriptors.reserveBuffer(vf.size() * 256);
    for (int i = 0; i < vf.size(); ++i) {
      auto f = vf[i];
      cv::KeyPoint kp =
          f->keypoint(); // preserve all the properties of the initial keypoint
      kp.pt.x = pts1[i].x; // with updated pixel location
      kp.pt.y = pts1[i].y;
      kp.class_id = i;
      kps.push_back(kp);
    }
    std::vector<cv::KeyPoint> kps_copy = kps;
    extractor_->compute(img_, kps, descriptors);
    if (kps_copy.size() > kps.size()) {
      MatchKeypointsAndStatus(kps_copy, kps, status);
    }

    for (int i = 0; i < kps.size(); ++i) {
      auto f = vf[kps[i].class_id];
      if (descriptor_distance_thresh_ != -1) {
        int dist =
            cv::norm(f->descriptor(), descriptors.row(i), cv::NORM_HAMMING);
        if (dist > descriptor_distance_thresh_) {
          status[i] = 0; // enforce to be dropped
        }
      }
      // FIXME: so if the distance test fails, we probably do not want to update
      // the descriptor
      // set new descriptor
      f->SetDescriptor(descriptors.row(i));
    }
  }

  // iterate through features and mark bad ones
  int num_valid_features = 0;
  int i = 0;

  // Clear list of newly dropped tracks from last time
  newly_dropped_tracks_.clear();

  // Update tracks
  for (auto it = features_.begin(); it != features_.end(); ++it, ++i) {
    FeaturePtr f(*it);

    Vec2 last_pos(f->xp());
    if (status[i]) {
      if (MaskValid(mask_, pts1[i].x, pts1[i].y) &&
          (last_pos - Vec2{pts1[i].x, pts1[i].y}).norm() <
              max_pixel_displacement_) {
        // FIXME: SUPER HACK drop features to enforce update
        // update track status
        f->SetTrackStatus(TrackStatus::TRACKED);
        f->UpdateTrack(pts1[i].x, pts1[i].y);
        // MaskOut(mask_, last_pos(0), last_pos(1), mask_size_);
        MaskOut(mask_, pts1[i].x, pts1[i].y, mask_size_);
        ++num_valid_features;
      } else {
        // failed to extract descriptors or invalid mask
        newly_dropped_tracks_.push_back(f);
        // MaskOut(mask_, last_pos(0), last_pos(1), mask_size_);
      }
    } else {
      // failed to track, reject
      newly_dropped_tracks_.push_back(f);
      // MaskOut(mask_, last_pos(0), last_pos(1), mask_size_);
    }
  }

  // detect a new set of features
  // this can rescue dropped featuers by matching them to newly detected ones
  if (num_valid_features < num_features_min_) {
    Detect(img_, num_features_max_ - num_valid_features);
  }

  // Mark all features that are still in newly_dropped_tracks_ at this point
  // as dropped
  for (auto f: newly_dropped_tracks_) {
    f->SetTrackStatus(TrackStatus::DROPPED);
  }

  // swap buffers ...
  std::swap(pyramid, pyramid_);

}

////////////////////////////////////////
// helpers
////////////////////////////////////////
void ResetMask(cv::Mat mask) { mask.setTo(255); }

void MaskOut(cv::Mat mask, number_t x, number_t y, int mask_size) {
  static int half_size = (mask_size >> 1);
  cv::rectangle(mask, cv::Point2d(x - half_size, y - half_size),
                cv::Point2d(x + half_size, y + half_size), cv::Scalar(0), -1);
}

bool MaskValid(const cv::Mat &mask, number_t x, number_t y) {
  int col = static_cast<int>(x);
  int row = static_cast<int>(y);
  if (col < 0 || col >= mask.cols || row < 0 || row >= mask.rows)
    return false;
  return static_cast<bool>(mask.at<uint8_t>(row, col));
}


void MatchKeypointsAndStatus(const std::vector<cv::KeyPoint> &orig_kps,
  const std::vector<cv::KeyPoint> &kps, std::vector<uint8_t> &status) {

  int n_diff = orig_kps.size() - kps.size();
  int n_found = 0;
  int orig_i = 0; // tracks indices in orig_kps
  int new_i = 0; // tracks indices in kps

  while (n_found < n_diff) {
    if (orig_kps[orig_i].class_id != kps[new_i].class_id) {
      status[orig_kps[orig_i].class_id] = 0;
      n_found++;
      orig_i++;
    } else {
      orig_i++;
      new_i++;
    }
  }
}


} // namespace xivo
