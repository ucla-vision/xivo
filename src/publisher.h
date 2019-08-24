// A wrapper of the viewer class to support threading.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "estimator_process.h"
#include "viewer.h"

namespace xivo {

class ViewMessage {
public:
  ViewMessage(const timestamp_t &ts) : ts_{ts} {}
  virtual void Execute(Viewer *) = 0;
  virtual ~ViewMessage() = default;

protected:
  timestamp_t ts_;
};

class ViewDisplayMessage : public ViewMessage {
public:
  ViewDisplayMessage(const timestamp_t &ts, const cv::Mat &image)
      : ViewMessage{ts}, image_{image.clone()} {}
  void Execute(Viewer *viewer) override {
    if (!image_.empty()) {
      viewer->Update(image_);
      viewer->Refresh();
    }
  }

private:
  cv::Mat image_;
};

class ViewPoseMessage : public ViewMessage {
public:
  ViewPoseMessage(const timestamp_t &ts, const SE3 &gsb, const SE3 &gbc)
      : ViewMessage{ts}, gsb_{gsb}, gbc_{gbc} {}

  void Execute(Viewer *viewer) override {
    viewer->Update_gsb(gsb_);
    viewer->Update_gsc(gsb_ * gbc_);
  }

private:
  SE3 gsb_, gbc_;
};

class ViewPublisher : public Publisher, public Process<ViewMessage> {
public:
  ViewPublisher(const Json::Value &cfg, const std::string &name = "",
                uint32_t size = 1000)
      : Process{size}, viewer_{cfg, name} {}
  virtual void Publish(const timestamp_t &ts, const cv::Mat &image) override;
  virtual void Publish(const timestamp_t &ts, const SE3 &gsb,
                       const SE3 &gbc) override;
  // void Spin(int seconds=0);

private:
  virtual bool Handle(ViewMessage *message) override;
  Viewer viewer_;
};
} // namespace xivo
