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

  auto klt_cfg = cfg_["KLT"];
  win_size_ = klt_cfg.get("win_size", 15).asInt();
  max_level_ = klt_cfg.get("max_level", 4).asInt();
  max_iter_ = klt_cfg.get("max_iter", 15).asInt();
  eps_ = klt_cfg.get("eps", 0.01).asDouble();

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

  // collect keypoints
  for (int i = 0; i < kps.size(); ++i) {
    const cv::KeyPoint &kp = kps[i];
    if (MaskValid(mask_, kp.pt.x, kp.pt.y)) {
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
  img_ = image.clone();
  if (cfg_.get("normalize", false).asBool()) {
    cv::normalize(image, img_, 0, 255, cv::NORM_MINMAX);
  }

  if (!initialized_) {
    rows_ = img_.rows;
    cols_ = img_.cols;
    mask_ = cv::Mat(rows_, cols_, CV_8UC1);
    mask_.setTo(0);

    // build image pyramid
    cv::buildOpticalFlowPyramid(img_, pyramid_, cv::Size(win_size_, win_size_),
                                max_level_);
    // setup the mask
    ResetMask(mask_(
        cv::Rect(margin_, margin_, cols_ - 2 * margin_, rows_ - 2 * margin_)));
    // detect an initial set of features
    Detect(img_, num_features_max_);
    initialized_ = true;
    // std::cout << "tracker initialized";
    return;
  }
  // reset mask
  ResetMask(mask_(
      cv::Rect(margin_, margin_, cols_ - 2 * margin_, rows_ - 2 * margin_)));
  // std::cout << "reset mask\n";

  // build new pyramid
  std::vector<cv::Mat> pyramid;
  cv::buildOpticalFlowPyramid(img_, pyramid, cv::Size(win_size_, win_size_),
                              max_level_);

  // prepare for optical flow
  cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                            max_iter_, eps_);

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
                           cv::Size(win_size_, win_size_), max_level_, criteria,
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
    extractor_->compute(img_, kps, descriptors);

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
        f->SetTrackStatus(TrackStatus::DROPPED);
        // MaskOut(mask_, last_pos(0), last_pos(1), mask_size_);
      }
    } else {
      // failed to track, reject
      f->SetTrackStatus(TrackStatus::DROPPED);
      // MaskOut(mask_, last_pos(0), last_pos(1), mask_size_);
    }
  }

  // detect a new set of features
  if (num_valid_features < num_features_min_) {
    Detect(img_, num_features_max_ - num_valid_features);
    // TODO: rescue dropped featuers by matching them to newly detected ones
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

} // namespace xivo
