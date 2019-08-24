// Drawing functions to overlay feature tracks & system
// info on input images.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <list>

#include "opencv2/core/core.hpp"

#include "core.h"

namespace xivo {

class Canvas;
using CanvasPtr = Canvas *;

class Canvas {
public:
  static CanvasPtr instance();

  static void Delete();
  void Update(const cv::Mat &img);
  void Draw(const FeaturePtr f);
  void OverlayStateInfo(const State &X, int vspace = 12, int hspace = 12,
                        int thickness = 1, double font_scale = 0.9);
  const cv::Mat &display() const { return disp_; }

private:
  Canvas(const Canvas &) = delete;
  Canvas &operator=(const Canvas &) = delete;
  Canvas() = default;
  static std::unique_ptr<Canvas> instance_;

  cv::Mat disp_;
};
}
