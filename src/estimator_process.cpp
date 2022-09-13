// 3rdparty
#include "opencv2/highgui/highgui.hpp"
// xivo
#include "estimator_process.h"
#include "tracker.h"
#include "visualize.h"
#include "core.h"

namespace xivo {

bool operator<(const std::unique_ptr<EstimatorMessage> &m1,
               const std::unique_ptr<EstimatorMessage> &m2) {
  return m1->ts() < m2->ts();
}

void EstimatorProcess::Initialize(const std::string &config_path) {
  auto est_cfg = LoadJson(config_path);
  // estimator_ = std::unique_ptr<Estimator>(new Estimator{est_cfg});
  estimator_ = CreateSystem(est_cfg);
}

/*
void EstimatorProcess::Wait() {
  auto msg = new Block();
  Enqueue(msg);
  while (!msg->ready_) {
    usleep(1);
  }
}
*/

bool EstimatorProcess::Handle(EstimatorMessage *message) {
  if (Process::Handle(message))
    return true;

  message->Execute(estimator_);

  if (auto msg = dynamic_cast<VisualMeas *>(message)) {
    // FIXME: instead of drawing on the canvas in the estimator
    // the estimator can pass state and feature tracks to the publisher
    // and let the publisher draw stuff.
    // However, the publisher should live in its own thread, otherwise,
    // the drawing is still performed in the current (estimator_process) thread,
    // and will not make it more efficient.
    if (msg->viz() && publisher_ != nullptr) {
      publisher_->Publish(msg->ts(), Canvas::instance()->display());
    }

    if (pose_publisher_ != nullptr) {
      MatX fullcov = estimator_->Pstate();
      Mat6 posecov = fullcov.block<6,6>(0,0);
      pose_publisher_->Publish(msg->ts(), estimator_->gsb(), posecov);
    }

    if (map_publisher_ != nullptr) {
      Eigen::Matrix<number_t, Eigen::Dynamic, 3> InstateXs;
      Eigen::Matrix<number_t, Eigen::Dynamic, 6> InstateCov;
      Eigen::Matrix<number_t, Eigen::Dynamic, 2> InstatePx;
      VecXi feature_ids;
      int npts;
      estimator_->InstateFeaturePositionsAndCovs(max_pts_to_publish_, npts,
        InstateXs, InstateCov, InstatePx, feature_ids);
      map_publisher_->Publish(msg->ts(), npts, InstateXs, InstateCov,
        InstatePx, feature_ids);
    }

    if (full_state_publisher_ != nullptr) {

      MatX CameraCov = estimator_->CameraCov();

      full_state_publisher_->Publish(
        msg->ts(),
        estimator_->X(),
        estimator_->Ca(),
        estimator_->Cg(),
        estimator_->Pstate(),
        estimator_->gauge_group(),
        estimator_->gsc(),
        CameraCov);
    }

    if (twod_nav_publisher_ != nullptr) {
      twod_nav_publisher_->Publish(msg->ts(), estimator_->gsb(),
        estimator_->Vsb(), estimator_->Rsg(), estimator_->Pstate());
    }

    return true;
  } else if (auto msg = dynamic_cast<InertialMeas *>(message)) {

    if (msg->viz() && publisher_ != nullptr) {
      publisher_->Publish(msg->ts(), estimator_->gsb(), estimator_->gbc());
    }
    return true;
  }
  return false;
}

} // namespace xivo
