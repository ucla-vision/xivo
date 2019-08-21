// Viewer for VIO.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

#include <array>
#include <string>

#include "opencv2/core/core.hpp"
#include "pangolin/pangolin.h"

#include "core.h"

namespace feh {

using XYZRGB = std::array<float, 6>;

class Viewer {
public:
  Viewer(const Json::Value &cfg, const std::string &name = "");
  ~Viewer();

  void Update_gsb(const SE3 &gsb);
  void Update_gbc(const SE3 &gbc);
  void Update_gsc(const SE3 &gsc);
  void Update(const cv::Mat &img);
  void Refresh();

private:
  std::string window_name_;
  pangolin::OpenGlRenderState *camera_state_;
  pangolin::OpenGlRenderState *image_state_;
  pangolin::GlTexture *texture_;
  Json::Value cfg_;

  // viewport attributes
  int height_, width_;
  Mat3 K_, Kinv_;
  number_t fx_, fy_, cx_, cy_;
  number_t znear_, zfar_;

  cv::Mat image_;

  SE3 Rg_, gsb_, gbc_, gsc_;
  std::vector<Vec3> trace_; // body frame trajectory

  float bg_color_[4]; // background color (rgba)
  static int counter_;
};

} // feh
