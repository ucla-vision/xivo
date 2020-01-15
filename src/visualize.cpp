// Drawing functions to overlay feature tracks & system
// info on input images.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "opencv2/imgproc/imgproc.hpp"

#include "feature.h"
#include "visualize.h"
#include "param.h"

namespace xivo {

std::unique_ptr<Canvas> Canvas::instance_ = nullptr;

// BGR
static cv::Scalar kColorPink(203, 192, 255);
static cv::Scalar kColorCyan(255, 192, 203);
static cv::Scalar kColorRed(0, 0, 255);
static cv::Scalar kColorGreen(0, 255, 0);
static cv::Scalar kColorYellow(0, 255, 255);
static cv::Scalar kColorBlue(255, 0, 0);
static cv::Scalar kColorLakeBlue(219, 152, 52);

CanvasPtr Canvas::instance() {
  if (instance_ == nullptr) {
    instance_ = std::unique_ptr<Canvas>(new Canvas());
  }
  return instance_.get();
}

void Canvas::Update(const cv::Mat &img) {
  if (img.empty()) {
    return;
  }
  if (img.channels() == 1) {
    cv::cvtColor(img, disp_, CV_GRAY2RGB);
  } else {
    img.copyTo(disp_);
  }
}

void Canvas::Draw(const FeaturePtr f) {
  if (disp_.empty()) {
    return;
  }

  bool use_debug_view{ParameterServer::instance()->get("use_debug_view", false).asBool()};
  bool draw_OOS{ParameterServer::instance()->get("draw_OOS", false).asBool()};

  auto pos(f->xp());
  cv::Scalar color;
  if (f->track_status() == TrackStatus::TRACKED
      && (f->instate() || draw_OOS) ) {

    Vec2 last_pos(f->front());
    for (auto pos : *f) {
      if (pos != f->xp()) {
        last_pos = pos;
      } else
        break;
    }

    // draw the trace
    cv::line(disp_, cv::Point2d(pos[0], pos[1]),
             cv::Point2d(last_pos[0], last_pos[1]), kColorYellow, 1);

    if (f->instate()) {
      cv::drawMarker(disp_, cv::Point2d(pos[0], pos[1]), kColorGreen,
                     cv::MARKER_CROSS, 10, 2);
    } else {
      cv::circle(disp_, cv::Point2d(pos[0], pos[1]), 2, kColorRed, -1);
    }
  } else if (f->track_status() == TrackStatus::REJECTED ||
             f->track_status() == TrackStatus::DROPPED) {

    if (use_debug_view) {
      cv::drawMarker(disp_, cv::Point2d(pos[0], pos[1]), kColorPink,
          cv::MARKER_TRIANGLE_UP, 20, 5);
    }

  } else if (f->track_status() == TrackStatus::CREATED) {
    cv::circle(disp_, cv::Point2d(pos[0], pos[1]), 3, kColorYellow, -1);
  } else {
    // LOG(WARNING) << "Feature status NOT recognized.";
  }

  if (use_debug_view) {
    // overwrite rejected features
    // if (f->status() == FeatureStatus::REJECTED_BY_TRACKER) {
    //   cv::drawMarker(disp_, cv::Point2d(pos[0], pos[1]), kColorPink,
    //   cv::MARKER_TRIANGLE_UP, 20, 5);
    // } else

    if (f->status() == FeatureStatus::REJECTED_BY_FILTER) {
      cv::drawMarker(disp_, cv::Point2d(pos[0], pos[1]), kColorCyan,
          cv::MARKER_DIAMOND, 20, 5);
    }
  }
}

void Canvas::OverlayStateInfo(const State &X, int vspace, int hspace,
                              int thickness, double font_scale) {
  if (disp_.empty()) {
    return;
  }

  bool print_bias_info{ParameterServer::instance()->get("print_bias_info", false).asBool()};

  int line_counter{0};

  cv::putText(disp_, StrFormat("Tsb=[%0.4f, %0.4f, %0.4f]", X.Tsb(0),
                                     X.Tsb(1), X.Tsb(2)),
              cv::Point(hspace, vspace * ++line_counter), CV_FONT_HERSHEY_PLAIN, font_scale,
              kColorLakeBlue, thickness);

  cv::putText(disp_, StrFormat("Vsb=[%0.4f, %0.4f, %0.4f]", X.Vsb(0),
                                     X.Vsb(1), X.Vsb(2)),
              cv::Point(hspace, vspace * ++line_counter), CV_FONT_HERSHEY_PLAIN, font_scale,
              kColorLakeBlue, thickness);

  auto Wsb{X.Rsb.log()};
  cv::putText(disp_, StrFormat("Wsb=[%0.4f, %0.4f, %0.4f]", Wsb(0),
                                     Wsb(1), Wsb(2)),
              cv::Point(hspace, vspace * ++line_counter), CV_FONT_HERSHEY_PLAIN, font_scale,
              kColorLakeBlue, thickness);

  cv::putText(disp_, StrFormat("Tbc=[%0.4f, %0.4f, %0.4f]", X.Tbc(0),
                                     X.Tbc(1), X.Tbc(2)),
              cv::Point(hspace, vspace * ++line_counter), CV_FONT_HERSHEY_PLAIN, font_scale,
              kColorLakeBlue, thickness);

  auto Wbc{X.Rbc.log()};
  cv::putText(disp_, StrFormat("Wbc=[%0.4f, %0.4f, %0.4f]", Wbc(0),
                                     Wbc(1), Wbc(2)),
              cv::Point(hspace, vspace * ++line_counter), CV_FONT_HERSHEY_PLAIN, font_scale,
              kColorLakeBlue, thickness);


  if (print_bias_info) {
    cv::putText(disp_, StrFormat("bg=[%0.4f, %0.4f, %0.4f]", X.bg(0),
          X.bg(1), X.bg(2)),
        cv::Point(hspace, vspace * ++line_counter), CV_FONT_HERSHEY_PLAIN, font_scale,
        kColorLakeBlue, thickness);

    cv::putText(disp_, StrFormat("ba=[%0.4f, %0.4f, %0.4f]", X.ba(0),
          X.ba(1), X.ba(2)),
        cv::Point(hspace, vspace * ++line_counter), CV_FONT_HERSHEY_PLAIN, font_scale,
        kColorLakeBlue, thickness);
  }

  cv::putText(disp_, StrFormat("td=%0.4f", X.td),
      cv::Point(hspace, vspace * ++line_counter), CV_FONT_HERSHEY_PLAIN, font_scale,
      kColorLakeBlue, thickness);


}

}
