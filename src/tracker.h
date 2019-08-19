// The feature tracking module;
// Multi-scale Lucas-Kanade tracker from OpenCV.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

#include <list>
#include <memory>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "json/json.h"

#include "core.h"

namespace feh {

class Tracker {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static TrackerPtr Create(const Json::Value &cfg);
  static TrackerPtr instance() { return instance_.get(); }

  void Update(const cv::Mat &img);

public:
  std::list<FeaturePtr> features_;

private:
  Tracker(const Tracker &other) = delete;
  Tracker &operator=(const Tracker &other) = delete;

  Tracker(const Json::Value &cfg);
  static std::unique_ptr<Tracker> instance_;

  // variables
  bool initialized_;
  Json::Value cfg_;
  int descriptor_distance_thresh_; // use this to verify feature tracking
  int max_pixel_displacement_;     // pixels shifted larger than this amount are
                                   // dropped

  cv::Mat img_;
  std::vector<cv::Mat> pyramid_;
  int rows_, cols_;

  // for the geneirc feature2d interface, see the following openc document:
  // https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html
  cv::Ptr<cv::Feature2D> detector_, extractor_;
  bool extract_descriptor_;

  cv::Mat mask_;
  int mask_size_;
  int margin_;

  // optical flow params
  int win_size_;
  int max_level_;
  int max_iter_;
  ftype eps_;

  // fast params
  int num_features_min_;
  int num_features_max_;

private:
  void Detect(const cv::Mat &img, int num_to_add);
};

// helpers
void ResetMask(cv::Mat mask);
void MaskOut(cv::Mat mask, ftype x, ftype y, int mask_size = 15);
bool MaskValid(const cv::Mat &mask, ftype x, ftype y);

} // namespace feh
