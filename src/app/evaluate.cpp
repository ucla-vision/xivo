#include <fstream>
#include <iostream>
#include <memory>
#include <tuple>

#include "absl/strings/str_format.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "alias.h"
#include "message_types.h"
#include "metrics.h"
#include "se3.h"
#include "tumvi.h"

// flags
DEFINE_string(root, "/home/feixh/Data/tumvi/exported/euroc/512_16/",
              "Root directory containing tumvi dataset folder.");
DEFINE_string(dataset, "tumvi", "euroc | tumvi");
DEFINE_string(seq, "room1", "Sequence of TUM VI benchmark to play with.");
DEFINE_int32(cam_id, 0, "Camera id.");
DEFINE_string(result, "out_state", "Path to the result file.");

// flags for evaluation
DEFINE_double(resolution, 0.001,
              "Asynchronized timestemps within this bound, in seconds, are "
              "paired and compared."); // do interpolation instead
DEFINE_double(RPE_interval, 1.0,
              "Interval, in seconds, over which to compute RPE.");

namespace feh {
std::vector<msg::Pose> LoadEstimatedState(const std::string &path) {
  std::ifstream istream(path, std::ios::in);
  int64_t ts;       // timestamp
  number_t x, y, z;    // translation
  number_t w0, w1, w2; // rotation in exponential coordinates
  std::vector<msg::Pose> out;
  while (istream >> ts >> x >> y >> z >> w0 >> w1 >> w2) {
    auto R = SO3::exp({w0, w1, w2});
    auto T = Vec3{x, y, z};
    out.push_back({ts, SE3{R, T}});
  }
  return out;
}
}

using namespace feh;

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::string image_dir, imu_dir, mocap_dir;
  std::tie(image_dir, imu_dir, mocap_dir) =
      GetDirs(FLAGS_dataset, FLAGS_root, FLAGS_seq, FLAGS_cam_id);
  std::unique_ptr<TUMVILoader> loader(new TUMVILoader{image_dir, imu_dir});

  auto traj_gt = loader->LoadGroundTruthState(mocap_dir);
  LOG(INFO) << "Ground truth loaded";

  std::vector<msg::Pose> traj_est = LoadEstimatedState(FLAGS_result);

  LOG(INFO) << "Estimated trajectory loaded";
  // Absolute Trajectory Error
  number_t ate;
  SE3 g_est_gt;
  std::tie(ate, g_est_gt) = ComputeATE(traj_est, traj_gt, FLAGS_resolution);
  LOG(INFO) << "ATE computed";

  // Relative Position Error
  number_t rpe_pos, rpe_rot;
  std::tie(rpe_pos, rpe_rot) =
      ComputeRPE(traj_est, traj_gt, FLAGS_RPE_interval, FLAGS_resolution);
  LOG(INFO) << "RPE computed";

  std::cout << absl::StreamFormat("ATE=%0.4f meters\n", ate);
  std::cout << absl::StreamFormat(
      "RPE @ %0.4f ms=[%0.4f meters, %0.4f degrees]\n",
      1000.0 * FLAGS_RPE_interval, rpe_pos, rpe_rot / M_PI * 180);

  // write aligned estimates to file.
  std::string output_path = FLAGS_result + "_aligned";
  std::ofstream ostream{output_path, std::ios::out};
  if (!ostream.is_open())
    LOG(FATAL) << "failed to open output file @ " << output_path;
  ;
  for (auto msg : traj_est) {
    auto aligned_g = g_est_gt.inv() * msg.g_;
    ostream << absl::StrFormat("%ld", msg.ts_) << " "
            << aligned_g.T().transpose() << " "
            << aligned_g.R().log().transpose() << std::endl;
  }
  ostream.close();
  LOG(INFO) << "aligned trajectory saved" << std::endl;
}
