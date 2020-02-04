#include <string>
#include "simple_node.h"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"

#include "xivo/FeatureData.h"
#include "xivo/FeatureMap.h"

#include "utils.h"

namespace xivo
{

ros::Time xivoTimestamp_to_rosTime(timestamp_t ts) {
  int nsec_total = ts.count();
  int nsec = nsec_total % 1000000000UL;
  int sec = nsec_total / 1000000000UL;
  ros::Time ros_ts(sec, nsec);
  return ros_ts;
}


void ROSPublisherAdapter::Publish(const timestamp_t &ts, const cv::Mat &disp) {
  if (!disp.empty()) {
    DLOG(INFO) << "Display image is ready";
    cv_bridge::CvImage cv_msg;
    // cv_msg.header.stamp    = msg->header.stamp;
    cv_msg.header.frame_id = "image";
    cv_msg.header.stamp = xivoTimestamp_to_rosTime(ts);
    cv_msg.encoding        = disp.channels() > 1 ? "bgr8" : "mono8";
    cv_msg.image           = disp;
    sensor_msgs::Image ros_msg;
    cv_msg.toImageMsg(ros_msg);
    rospub_.publish(ros_msg);
    cv_msg.image.release();
  }
}


void ROSEgoMotionPublisherAdapter::Publish(const timestamp_t &ts,
  const SE3 &gsb, const Mat6 &cov)
{
  geometry_msgs::PoseWithCovarianceStamped msg;
  msg.header.frame_id = "Robot State";
  msg.header.stamp = xivoTimestamp_to_rosTime(ts);

  Vec3 pos = gsb.translation();
  msg.pose.pose.position.x = pos(0);
  msg.pose.pose.position.y = pos(1);
  msg.pose.pose.position.z = pos(2);

  Mat3 rot = gsb.rotation();
  Quat q(rot);
  msg.pose.pose.orientation.x = q.x();
  msg.pose.pose.orientation.y = q.y();
  msg.pose.pose.orientation.z = q.z();
  msg.pose.pose.orientation.w = q.w();

  // Row-major covariance
  int count = 0;
  for (int i=0; i<6; i++) {
    for (int j=0; j<6; j++) {
      msg.pose.covariance[count] = cov(i,j);
      count++;
    }
  }

  rospub_.publish(msg);
}


void ROSMapPublisherAdapter::Publish(number_t ts, const int num_features,
  const VecX &poses, const MatX &covs)
{
  FeatureMap msg;
  msg.header.frame_id = "Feature Map";
  msg.header.stamp = ros::Time(ts);

  msg.num_features = (unsigned int) num_features;

  for (int i=0; i<num_features; i++) {
    FeatureData f;
  }
}



SimpleNode::~SimpleNode() {
  if (est_proc_) {
    est_proc_->Wait();
    est_proc_.reset();
  }
  if (adapter_) {
    adapter_.reset();
  }
  if (viewer_) {
    viewer_->Wait();
    viewer_.reset();
  }
}

SimpleNode::SimpleNode(): adapter_{nullptr}, viewer_{nullptr}, viz_{false}
{
  // parse node parameters
  ros::NodeHandle nh_priv("~");

  // get configuration path & create estimator process
  std::string cfg_path;
  int estimator_queue_size;

  nh_priv.param("config_path", cfg_path, cfg_path);
  LOG(INFO) << "Found configuration @ " << cfg_path;

  // get image and imu topics
  std::string imu_topic, image_topic;
  nh_priv.param("imu_topic", imu_topic, imu_topic);
  nh_priv.param("image_topic", image_topic, image_topic);

  // subscribe to topics
  imu_sub_ = nh_.subscribe(imu_topic, 1000, &SimpleNode::ImuMsgCallback, this);
  LOG(INFO) << "Subscribed to IMU topic: " << imu_topic;

  image_sub_ = nh_.subscribe(image_topic, 1000, &SimpleNode::ImageMsgCallback, this);
  LOG(INFO) << "Subscribed to image topic: " << image_topic;

  // estimator process
  nh_priv.param("estimator_queue_size", estimator_queue_size, estimator_queue_size);
  LOG(INFO) << "Size of estimator process queue = " << estimator_queue_size;

  est_proc_ = std::unique_ptr<EstimatorProcess>(
      new EstimatorProcess("Estimator", estimator_queue_size));
  est_proc_->Initialize(cfg_path);

  std::string viewer_type{"ros"};
  nh_priv.param("viewer_type", viewer_type, viewer_type);
  if (viewer_type.empty()) {
    LOG(WARNING) << "viewer type not set!";
  } else {
    viz_ = true;
    if (viewer_type == "ros") {
      // publisher
      viz_pub_ = nh_.advertise<sensor_msgs::Image>("TrackerView", 1);
      adapter_ = std::unique_ptr<ROSPublisherAdapter>(
          new ROSPublisherAdapter(viz_pub_));
      // set publisher to estimator process
      est_proc_->SetPublisher(adapter_.get());
    } else if (viewer_type == "pangolin") {
      std::string viewer_cfg_path;
      nh_priv.param("viewer_config_path", viewer_cfg_path, viewer_cfg_path);
      auto viewer_cfg = LoadJson(viewer_cfg_path);
      viewer_ = std::unique_ptr<ViewPublisher>(
          new ViewPublisher(viewer_cfg));
      // start the view-publisher thread
      viewer_->Start();
      // set publisher to estimator process
      est_proc_->SetPublisher(viewer_.get());
    } else {
      LOG(FATAL) << "unknown viewer type; only support ros & pangolin (all lower case)";
    }
  }

  nh_priv.param("publish_state", publish_egomotion_, true);
  nh_priv.param("publish_map", publish_map_, true);
  if (publish_egomotion_) {
    std::cout << "publishing egomotion" << std::endl;
    ego_motion_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(
      "xivo/egomotion", 1000);
    ego_motion_adapter_ = std::unique_ptr<ROSEgoMotionPublisherAdapter>(
      new ROSEgoMotionPublisherAdapter(ego_motion_pub_));
    est_proc_->SetPosePublisher(ego_motion_adapter_.get());
  }
  if (publish_map_) {
    std::cout << "publishing map" << std::endl;
    map_pub_ = nh_.advertise<xivo::FeatureMap>("xivo/map", 1000);
    map_adapter_ = std::unique_ptr<ROSMapPublisherAdapter>(
      new ROSMapPublisherAdapter(map_pub_));
    est_proc_->SetMapPublisher(map_adapter_.get());
  }


  // start the estimator thread
  est_proc_->Start();
}

void SimpleNode::ImuMsgCallback(sensor_msgs::ImuConstPtr msg)
{
  Vec3 gyro{msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z};
  Vec3 accel{msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z};
  timestamp_t ts{msg->header.stamp.toNSec()};

  est_proc_->Enqueue(std::move(std::make_unique<InertialMeas>(ts, gyro, accel, viz_)));
}

void SimpleNode::ImageMsgCallback(sensor_msgs::ImageConstPtr msg)
{
  timestamp_t ts{msg->header.stamp.toNSec()};
  cv_bridge::CvImagePtr cv_ptr;
  cv::Mat image;
  if (msg->encoding == sensor_msgs::image_encodings::MONO8) {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    cv_ptr->image.copyTo(image);
  } else if (msg->encoding == sensor_msgs::image_encodings::MONO16) {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO16);
    cv_ptr->image.convertTo(image, CV_8UC1, 1.0/256);
  } else if (msg->encoding == sensor_msgs::image_encodings::RGB8) {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    cv::cvtColor(cv_ptr->image, image, CV_RGB2GRAY);
  } else if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::cvtColor(cv_ptr->image, image, CV_BGR2GRAY);
  } else {
    LOG(FATAL) << "unexpected image encoding";
  }

  est_proc_->Enqueue(std::move(std::make_unique<VisualMeas>(ts, image, viz_)));
}

} // namespace xivo

