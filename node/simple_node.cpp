#include <string>
#include "simple_node.h"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"

#include "xivo/FeatureData.h"
#include "xivo/FeatureMap.h"
#include "xivo/FullState.h"

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



template<class T> // T is either geometry_msgs/Vector3 or geometry_msgs/Point
void copy_vec3_to_ros(T &ros_dest, const Vec3 &vector)
{
  ros_dest.x = vector(0);
  ros_dest.y = vector(1);
  ros_dest.z = vector(2);
}


void copy_rot_to_ros(geometry_msgs::Quaternion &ros_quat, const SO3 &xivo_rot)
{
  Mat3 R = xivo_rot.matrix();
  Quat q(R);
  ros_quat.x = q.x();
  ros_quat.y = q.y();
  ros_quat.z = q.z();
  ros_quat.w = q.w();
}


template<class T, unsigned long l>
void copy_full_square_mat_to_ros(boost::array<double, l> &rosarr, T &matrix, int dim)
{
  int count = 0;
  for (int i=0; i<dim; i++) {
    for (int j=0; j<dim; j++) {
      rosarr[count] = matrix(i,j);
      count++;
    }
  }
}

template<class T, unsigned long l>
void copy_upper_triangle_to_ros(boost::array<double, l> &rosarr, T matrix, int dim)
{
  int count = 0;
  for (int i=0; i<dim; i++) {
    for (int j=i; j<dim; j++) {
      rosarr[count] = matrix(i,j);
      count++;
    }
  }
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

  copy_vec3_to_ros(msg.pose.pose.position, gsb.translation());
  // Quaternion in pose is inverse of quaternion in transformation
  copy_rot_to_ros(msg.pose.pose.orientation, gsb.rotation().inv());

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


void ROSMapPublisherAdapter::Publish(const timestamp_t &ts, const int npts,
  const VecX &InstateXs, const MatX &InstateCov, const VecXi &feature_ids)
{
  FeatureMap msg;
  msg.header.frame_id = "Feature Map";
  msg.header.stamp = xivoTimestamp_to_rosTime(ts);
  msg.num_features = npts;

  for (int i=0; i<npts; i++) {
    FeatureData f;
    f.id = feature_ids(i);

    f.Xs.x = InstateXs(i,0);
    f.Xs.y = InstateXs(i,1);
    f.Xs.z = InstateXs(i,2);

    f.covariance[0] = InstateCov(i,0);
    f.covariance[1] = InstateCov(i,1);
    f.covariance[2] = InstateCov(i,2);

    f.covariance[3] = InstateCov(i,1);
    f.covariance[4] = InstateCov(i,3);
    f.covariance[5] = InstateCov(i,4);

    f.covariance[6] = InstateCov(i,2);
    f.covariance[7] = InstateCov(i,4);
    f.covariance[8] = InstateCov(i,5);

    msg.features.push_back(f);
  }

  rospub_.publish(msg);
}


void ROSFullStatePublisherAdapter::Publish(const timestamp_t &ts,
  const State &X, const Mat3 &Ca, const Mat3 &Cg, const MatX &Cov)
{
  FullState msg;
  msg.header.frame_id = "full state";
  msg.header.stamp = xivoTimestamp_to_rosTime(ts);

  copy_vec3_to_ros(msg.gsb.translation, X.Tsb);
  copy_rot_to_ros(msg.gsb.rotation, X.Rsb);
  copy_vec3_to_ros(msg.Vsb, X.Vsb);
  copy_vec3_to_ros(msg.gbc.translation, X.Tbc);
  copy_rot_to_ros(msg.gbc.rotation, X.Rbc);
  copy_vec3_to_ros(msg.bg, X.bg);
  copy_vec3_to_ros(msg.ba, X.ba);
  copy_rot_to_ros(msg.qg, X.Rg);

  msg.td = X.td;

  copy_full_square_mat_to_ros(msg.Cg, Cg, 3);
  copy_upper_triangle_to_ros(msg.Ca, Ca, 3);

  rospub_.publish(msg);
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

  int max_features_to_publish;
  nh_priv.param("publish_state", publish_egomotion_, false);
  nh_priv.param("publish_full_state", publish_full_state_, false);
  nh_priv.param("publish_map", publish_map_, false);
  nh_priv.param("max_features_to_publish", max_features_to_publish, 100);
  if (publish_egomotion_) {
    ego_motion_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(
      "xivo/pose", 1000);
    ego_motion_adapter_ = std::unique_ptr<ROSEgoMotionPublisherAdapter>(
      new ROSEgoMotionPublisherAdapter(ego_motion_pub_));
    est_proc_->SetPosePublisher(ego_motion_adapter_.get());
  }
  if (publish_map_) {
    map_pub_ = nh_.advertise<xivo::FeatureMap>("xivo/map", 1000);
    map_adapter_ = std::unique_ptr<ROSMapPublisherAdapter>(
      new ROSMapPublisherAdapter(map_pub_));
    est_proc_->SetMapPublisher(map_adapter_.get(), max_features_to_publish);
  }
  if (publish_full_state_) {
    full_state_pub_ = nh_.advertise<xivo::FullState>("xivo/fullstate", 1000);
    full_state_adapter_ = std::unique_ptr<ROSFullStatePublisherAdapter>(
      new ROSFullStatePublisherAdapter(full_state_pub_));
    est_proc_->SetFullStatePublisher(full_state_adapter_.get());
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

