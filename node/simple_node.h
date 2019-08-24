#pragma once
#include <vector>
#include <memory>

#include "Eigen/Core"
#include "ros_setup.h"

#include "estimator_process.h"
#include "publisher.h"

namespace xivo
{

// Size of the small buffer used for sorting messages based on timestamps.
constexpr int ros_msg_buf_size = 10;

class ROSPublisherAdapter: public Publisher {
public:
  ROSPublisherAdapter(ros::Publisher &rospub): Publisher{}, rospub_{rospub}
  {}
  virtual void Publish(ftype ts, const cv::Mat &disp);
private:
  ros::Publisher &rospub_;
};

class SimpleNode
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SimpleNode();
  ~SimpleNode();
  void ImuMsgCallback(sensor_msgs::ImuConstPtr msg);
  void ImageMsgCallback(sensor_msgs::ImageConstPtr msg);
private:
  std::unique_ptr<EstimatorProcess> est_proc_;
  ros::NodeHandle nh_;
  ros::Subscriber imu_sub_, image_sub_;
  // std::priority_queue<std::unique_ptr<EstimatorMessage>> queue_;
  std::vector<std::unique_ptr<EstimatorMessage>> buf_;

  bool viz_;  // enable vizualization if true
  // the following two (viz_pub and adapter) are used together
  ros::Publisher viz_pub_;
  std::unique_ptr<ROSPublisherAdapter> adapter_;  // to be shared with the estimator process

  std::unique_ptr<ViewPublisher> viewer_; // pangolin-based view publisher
};

}
