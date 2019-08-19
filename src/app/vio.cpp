// Visual-Inertial Odometry application with threading.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "unistd.h"
#include <algorithm>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"

#include "estimator.h"
#include "estimator_process.h"
#include "metrics.h"
#include "tracker.h"
#include "tumvi.h"
#include "publisher.h"
#include "visualize.h"

// flags
DEFINE_string(cfg, "cfg/vio.json",
              "Configuration file for the VIO application.");
DEFINE_string(root, "/home/feixh/Data/tumvi/exported/euroc/512_16/",
              "Root directory containing tumvi dataset folder.");
DEFINE_string(dataset, "tumvi", "euroc | tumvi");
DEFINE_string(seq, "room1", "Sequence of TUM VI benchmark to play with.");
DEFINE_int32(cam_id, 0, "Camera id.");
DEFINE_string(out, "out_state", "Output file path.");

using namespace feh;

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto cfg = LoadJson(FLAGS_cfg);
  bool verbose = cfg.get("verbose", false).asBool();
  bool viz = cfg.get("visualize", false).asBool();

  std::string image_dir, imu_dir, mocap_dir;
  std::tie(image_dir, imu_dir, mocap_dir) =
      GetDirs(FLAGS_dataset, FLAGS_root, FLAGS_seq, FLAGS_cam_id);

  TUMVILoader loader(image_dir, imu_dir);

  // create estimator
  std::unique_ptr<EstimatorProcess> est_proc(
      new EstimatorProcess{"Estimator", 1000});
  est_proc->Initialize(cfg["estimator_cfg"].asString());

  // create view publisher
  std::unique_ptr<ViewPublisher> publisher;
  if (viz) {
    auto viewer_cfg = LoadJson(cfg["viewer_cfg"].asString());
    publisher = std::unique_ptr<ViewPublisher>(
        new ViewPublisher(viewer_cfg));
    est_proc->SetPublisher(publisher.get());
    publisher->Start();
  }
  est_proc->Start();

  for (int i = 0; i < loader.size(); ++i) {
    if (verbose && i % 1000 == 0) {
      std::cout << i << "/" << loader.size() << std::endl;
    }
    auto msg = loader.Get(i);

    if (typeid(*msg) == typeid(msg::Image)) {

      auto image_msg = dynamic_cast<msg::Image *>(msg);
      auto image = cv::imread(image_msg->image_path_);
      est_proc->Enqueue(std::move(
            std::make_unique<VisualMeas>(image_msg->ts_, image, viz)));

    } else if (typeid(*msg) == typeid(msg::IMU)) {

      auto imu_msg = dynamic_cast<msg::IMU *>(msg);
      est_proc->Enqueue(std::move(
          std::make_unique<InertialMeas>(imu_msg->ts_, imu_msg->gyro_, imu_msg->accel_, viz)));

    } else {
      throw std::runtime_error("Invalid entry type.");
    }
    ////////////////////////////////////////
    // EXAMPLE: SYNC AND READ OUT CURRENT ESTIMATES
    ////////////////////////////////////////
    if (verbose) {
      est_proc->Wait();
      std::cout << absl::StrFormat("%ld", est_proc->ts().count()) << " "
        << est_proc->gsb().translation().transpose() << std::endl;
    }
  }
  est_proc->Wait();
  // sleep(5);
  std::cout << "done" << std::endl;
  est_proc.reset();
  std::cout << "estimator-process reset" << std::endl;
  if (publisher) {
    publisher->Wait();
    publisher.reset();
    std::cout << "publisher reset" << std::endl;
  }
}
