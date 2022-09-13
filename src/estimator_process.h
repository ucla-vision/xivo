// A process where the estimator resides.
// Also allows the estimator to talk publishers.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
// stl
#include <fstream>
#include <string>
#include <vector>
// xivo
#include "alias.h"
#include "estimator.h"
#include "message_types.h"
#include "process.h"

namespace xivo {

class Publisher {
public:
  virtual ~Publisher(){};
  virtual void Publish(const timestamp_t &ts, const cv::Mat &image) {}
  virtual void Publish(const timestamp_t &ts, const SE3 &gsb, const SE3 &gbc) {}
  virtual void Publish(const timestamp_t &ts, const SE3 &gsb, const Mat6 &cov) {}
  virtual void Publish(const timestamp_t &ts, const int npts,
    const Eigen::Matrix<number_t, Eigen::Dynamic, 3> &poses,
    const Eigen::Matrix<number_t, Eigen::Dynamic, 6> &covs,
    const Eigen::Matrix<number_t, Eigen::Dynamic, 2> &pixels,
    const VecXi &feature_ids) {}
  virtual void Publish(const timestamp_t &ts, const State &X,
    const Mat3 &Ca, const Mat3 &Cg, const MatX &Cov,
    const int gauge_group, const SE3 &gsc, const MatX &CameraCov) {}
  virtual void Publish(const timestamp_t &ts, const SE3 &gsb, const Vec3 &Vsb,
    const SO3 &Rg, const MatX &Cov) {}
};

class EstimatorMessage {
public:
  EstimatorMessage(const timestamp_t &ts, bool viz = false)
      : ts_{ts}, viz_{viz} {}

  timestamp_t ts() const { return ts_; }
  bool viz() const { return viz_; }

  virtual void Execute(Estimator *) = 0;
  virtual ~EstimatorMessage() {}

protected:
  timestamp_t ts_;
  bool viz_; // if set, visualize the filtering result
};

bool operator<(const std::unique_ptr<EstimatorMessage> &m1,
               const std::unique_ptr<EstimatorMessage> &m2);

class VisualMeas : public EstimatorMessage {
public:
  VisualMeas(const timestamp_t &ts, cv::Mat image, bool viz = false)
      : EstimatorMessage{ts, viz}, image_{image} {}

  void Execute(Estimator *est) override { est->VisualMeas(ts_, image_); }

private:
  cv::Mat image_;
};

class InertialMeas : public EstimatorMessage {
public:
  InertialMeas(const timestamp_t &ts, const Vec3 &gyro, const Vec3 &accel,
               bool viz = false)
      : EstimatorMessage{ts, viz}, gyro_{gyro}, accel_{accel} {}

  void Execute(Estimator *est) override {
    est->InertialMeas(ts_, gyro_, accel_);
  }

private:
  Vec3 gyro_, accel_;
};

class EstimatorProcess : public Process<EstimatorMessage> {
public:
  EstimatorProcess(const std::string &name, uint32_t size = 1000)
      : Process{size}, name_{name}, estimator_{nullptr}, publisher_{nullptr} {
    LOG(INFO) << "Process " << name_ << " created!";
  }
  void Initialize(const std::string &config_path);
  void SetPublisher(Publisher *publisher) { publisher_ = publisher; }
  void SetPosePublisher(Publisher *publisher) { pose_publisher_ = publisher; }
  void SetMapPublisher(Publisher *publisher, int max_pts_to_publish) {
    max_pts_to_publish_ = max_pts_to_publish;
    map_publisher_ = publisher;
  }
  void SetFullStatePublisher(Publisher * publisher) {
    full_state_publisher_ = publisher;
  }
  void Set2dNavStatePublisher(Publisher *publisher) {
    twod_nav_publisher_ = publisher;
  }

  ////////////////////////////////////////
  // used for synchronized communication
  // call wait first, and then call the available accessors
  ////////////////////////////////////////
  cv::Mat display() { return Canvas::instance()->display(); }
  // only call the following accessors when the process is properly synced
  SE3 gsb() const { return estimator_->gsb(); }
  SE3 gbc() const { return estimator_->gbc(); }
  SE3 gsc() const { return estimator_->gsc(); }
  State state() const { return estimator_->X(); }
  const timestamp_t &ts() const { return estimator_->ts(); }
  ////////////////////////////////////////

private:
  virtual bool Handle(EstimatorMessage *message) override;

private:
  std::string name_;
  // std::unique_ptr<Estimator> estimator_; // owned
  EstimatorPtr estimator_; // owned
  // results publisher for asynchronized communication for viewer
  Publisher *publisher_; // non-owned
  // results publisher for asynchronized communication for pose and map.
  Publisher *pose_publisher_;
  Publisher *map_publisher_;
  Publisher *full_state_publisher_;
  Publisher *twod_nav_publisher_;
  int max_pts_to_publish_;
};                       // EstimatorProcess

} // namespace xivo
