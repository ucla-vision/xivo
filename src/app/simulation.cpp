#include <fstream>
#include <iomanip>
#include <memory>

#include "absl/strings/str_format.h"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"
#include "json/json.h"

#include "core.h"
#include "estimator.h"
#include "simulator.h"

using namespace feh;

// \brief: Clone state from simulator to estimator for initialization in
// simulation.
void CloneState(std::shared_ptr<Estimator> est, std::shared_ptr<Simulator> sim,
                bool add_noise = false) {
  // clone nominal state
  est->X_ = sim->X_;
  // clone imu calibration
  est->Cas_ = sim->Cas_;
  est->Car_ = sim->Car_;
  est->Cgs_ = sim->Cgs_;
  est->Cgr_ = sim->Cgr_;
  // clone gravity
  est->g_ = sim->g_;

  // now add some noise to initial state
  if (add_noise) {
    auto n = sim->cfg_["init_noise_std"];
    est->X_.Rsb =
        SO3::exp(est->X_.Rsb.log() + RandomVector<3>(0., n["W"].asDouble()));
    est->X_.Tsb += RandomVector<3>(0., n["T"].asDouble());
    est->X_.Vsb += RandomVector<3>(0., n["V"].asDouble());
    est->X_.bg += RandomVector<3>(0., n["bg"].asDouble());
    est->X_.ba += RandomVector<3>(0., n["ba"].asDouble());
    est->X_.Rbc =
        SO3::exp(est->X_.Rbc.log() + RandomVector<3>(0., n["Wbc"].asDouble()));
    est->X_.Tbc += RandomVector<3>(0., n["Tbc"].asDouble());
    Vec3 Wg = est->X_.Rg.log();
    Wg.head<2>() += RandomVector<2>(0., n["Wg"].asDouble());
    Wg(2) = 0;
    est->X_.Rg = SO3::exp(Wg);
  }

  auto dX = est->X_ - sim->X_;
  LOG(INFO) << "Estimator state initialized from simulator.";
  LOG(INFO) << absl::StrFormat("||dT||=%0.2f, ||dR||=%0.2f\n", dX.Tsb.norm(),
                               dX.Rsb.log().norm());
}

//############################################################################
int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  std::string cfg_path;
  if (argc < 2) {
    cfg_path = "../cfg/simulation.json";
    LOG(INFO) << "Usage: ./simulate [CONFIGURATION]" << std::endl;
    LOG(INFO) << "CONFIGURATION not specified; use default configuration @ "
              << cfg_path << std::endl;
  } else {
    cfg_path = argv[1];
  }

  auto cfg = LoadJson(cfg_path);

  auto est = std::make_shared<Estimator>(cfg["estimator_cfg"].asString());
  auto sim = std::make_shared<Simulator>(cfg["simulator_cfg"].asString());
  // duplicate the initial state ...
  // otherwise, this won't work since the initial gravity rotation is not right
  // and we are not integrating gravity properly
  CloneState(est, sim, cfg.get("add_noise_to_init_state", false).asBool());

  msg::IMU imu_msg;
  msg::Track track_msg;

  imu_msg.initialized_ = false;
  track_msg.initialized_ = false;

  bool enable_update = cfg.get("enable_update", true).asBool();

  cv::Mat disp;
  ftype path_length(0);
  int counter(0);
  State last_X;
  while (sim->GetMeas(imu_msg, track_msg)) {
    if (imu_msg.initialized_) {
      est->InertialMeas(imu_msg.ts_, imu_msg.gyro_, imu_msg.accel_);
      if (!enable_update) {
        // pretend vision module has initialized
        est->vision_initialized_ = true;
        est->curr_time_ = imu_msg.ts_;
      }
    }

    if (track_msg.initialized_) {
      if (enable_update) {
        est->ProcessTracks_EKF(track_msg.ts_, track_msg.features_);
      }
      disp = sim->VisualizeTracks();
    }
    auto dX = est->X_ - sim->X_;
    VLOG(0) << absl::StrFormat("||dT||=%0.6f, ||dR||=%0.6f\n", dX.Tsb.norm(),
                               dX.Rsb.log().norm());
    VLOG(0) << absl::StrFormat("||dTbc||=%0.6f, ||dRbc||=%0.6f\n",
                               dX.Tbc.norm(), dX.Rbc.log().norm());
    if (counter != 0) {
      path_length += (sim->X_ - last_X).Tsb.norm();
      VLOG(0) << absl::StrFormat("T.drift=%0.2f%% of %0.2f meters\n",
                                 dX.Tsb.norm() / path_length * 100,
                                 path_length);
    }
    std::cout << path_length << " " << dX.Tsb.norm() << std::endl;
    last_X = sim->X_;
    ++counter;

    // reset messages
    imu_msg.initialized_ = false;
    track_msg.initialized_ = false;

    if (cfg.get("visualize", false).asBool()) {
      cv::imshow("simulation", disp);
      char ckey = cv::waitKey(5);
      if (ckey == 'q')
        break;
    }
  }
}
