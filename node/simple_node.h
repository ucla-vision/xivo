#pragma once
#include <vector>
#include <memory>

#include "Eigen/Core"
#include "alias.h"
#include "ros_setup.h"

#include "estimator_process.h"
#include "publisher.h"

namespace xivo
{

// Size of the small buffer used for sorting messages based on timestamps.
constexpr int ros_msg_buf_size = 10;

ros::Time xivoTimestamp_to_rosTime(timestamp_t ts);


// For publishing images to a ROS viewer
class ROSPublisherAdapter: public Publisher {
public:
  ROSPublisherAdapter(ros::Publisher &rospub): Publisher{}, rospub_{rospub}
  {}
  void Publish(const timestamp_t &ts, const cv::Mat &disp) override;
private:
  ros::Publisher &rospub_;
};


class ROSEgoMotionPublisherAdapter: public Publisher {
public:
  ROSEgoMotionPublisherAdapter(ros::Publisher &rospub): Publisher{}, rospub_{rospub}
  {}
  void Publish(const timestamp_t &ts, const SE3 &gsb, const Mat6 &cov) override;
private:
  ros::Publisher &rospub_;
};


class ROSMapPublisherAdapter: public Publisher {
public:
  ROSMapPublisherAdapter(ros::Publisher &rospub): Publisher{}, rospub_{rospub}
  {}
  void Publish(const number_t ts, const int num_features, const VecX &poses,
    const MatX &covs);
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

  bool publish_egomotion_; // Publishes state and covariance if true
  bool publish_map_; // Publishes instate features and covariances if true
  ros::Publisher ego_motion_pub_;
  ros::Publisher map_pub_;
  std::unique_ptr<ROSEgoMotionPublisherAdapter> ego_motion_adapter_;
  std::unique_ptr<ROSMapPublisherAdapter> map_adapter_;
};

}
