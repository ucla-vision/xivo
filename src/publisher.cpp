// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "publisher.h"

namespace feh {

void ViewPublisher::Publish(const timestamp_t &ts, const cv::Mat &image) {
  Enqueue(std::move(std::make_unique<ViewDisplayMessage>(ts, image)));
}

void ViewPublisher::Publish(const timestamp_t &ts, const SE3 &gsb,
                            const SE3 &gbc) {
  Enqueue(std::move(std::make_unique<ViewPoseMessage>(ts, gsb, gbc)));
}

/*
void ViewPublisher::Spin(int seconds) {
  int refresh_rate = 30;  // milliseconds
  // if seconds < 0, spin forever
  for (int i = 0; i < seconds * 1000 / refresh_rate; i += (seconds < 0 ? 0 : 1))
{
    Refresh();
    usleep(refresh_rate);
  }
}
*/

bool ViewPublisher::Handle(ViewMessage *message) {
  if (Process::Handle(message))
    return true;
  message->Execute(&viewer_);
  return true;
}

} // namespace feh
