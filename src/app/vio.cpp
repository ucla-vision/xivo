// Author: Xiaohan Fei
#include "unistd.h"
#include <algorithm>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"

#include "estimator.h"
#include "estimator_process.h"
#include "metrics.h"
#include "tracker.h"
#include "loader.h"
#include "viewer.h"
#include "visualize.h"
#include "graphwriter.h"
#include "graph.h"
#include "mapper.h"

// flags
DEFINE_string(cfg, "cfg/vio.json",
              "Configuration file for the VIO application.");
DEFINE_string(root, "/home/feixh/Data/tumvi/exported/euroc/512_16/",
              "Root directory containing tumvi dataset folder.");
DEFINE_string(dataset, "tumvi", "xivo | euroc | tumvi");
DEFINE_string(seq, "room1", "Sequence of TUM VI benchmark to play with.");
DEFINE_int32(cam_id, 0, "Camera id.");
DEFINE_string(out, "out_state", "Output file path.");
DEFINE_string(graphout, "", ".dot file to save output graph to");

using namespace xivo;


int GetLoopClosureInputs(std::vector<FastBrief::TDescriptor>& descriptors,
                         std::vector<cv::KeyPoint>& kps) {
  Graph& graph{*Graph::instance()};
  Mapper& mapper{*Mapper::instance()};

  std::vector<FeaturePtr> instate_features = graph.GetInstateFeatures();
  int num_instate_features = instate_features.size();

  if (num_instate_features > 0) {
    descriptors.reserve(num_instate_features);
    kps.reserve(num_instate_features);

    for (int i=0; i<num_instate_features; i++) {
      FeaturePtr f = instate_features[i];
      kps.push_back(f->keypoint());
      descriptors.push_back((FastBrief::TDescriptor)(f->descriptor().data));
    }
  }

  return instate_features.size();
}



int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto cfg = LoadJson(FLAGS_cfg);
  bool verbose = cfg.get("verbose", false).asBool();

  std::string image_dir, imu_dir, mocap_dir;
  std::tie(image_dir, imu_dir, mocap_dir) =
      GetDirs(FLAGS_dataset, FLAGS_root, FLAGS_seq, FLAGS_cam_id);

  std::unique_ptr<DataLoader> loader(new DataLoader{image_dir, imu_dir});

  // create estimator
  // auto est = std::make_unique<Estimator>(
  //     LoadJson(cfg["estimator_cfg"].asString()));
  auto est = CreateSystem(
      LoadJson(cfg["estimator_cfg"].asString()));

  // create viewer
  std::unique_ptr<Viewer> viewer;
  if (cfg.get("visualize", false).asBool()) {
    viewer = std::make_unique<Viewer>(
        LoadJson(cfg["viewer_cfg"].asString()), FLAGS_seq);
  }

  // Get mapper
  MapperPtr mapper = Mapper::instance();

  // setup I/O for saving results
  if (std::ofstream ostream{FLAGS_out, std::ios::out}) {

    std::vector<msg::Pose> traj_est;

    for (int i = 0; i < loader->size(); ++i) {
      auto raw_msg = loader->Get(i);

      if (verbose && i % 1000 == 0) {
        std::cout << i << "/" << loader->size() << std::endl;
      }

      if (auto msg = dynamic_cast<msg::Image *>(raw_msg)) {
        auto image = cv::imread(msg->image_path_);
        est->VisualMeas(msg->ts_, image);

        if (mapper->UseLoopClosure()) {
          std::vector<FastBrief::TDescriptor> descriptors;
          std::vector<cv::KeyPoint> kps;
          int num_feats = GetLoopClosureInputs(descriptors, kps);
          if (num_feats > 0) {
            Mapper::instance()->DetectLoopClosures(descriptors, kps);
          }
        }

        if (viewer) {
          viewer->Update_gsb(est->gsb());
          viewer->Update_gsc(est->gsc());

          cv::Mat disp = Canvas::instance()->display();

          if (!disp.empty()) {
            LOG(INFO) << "Display image is ready";
            viewer->Update(disp);
            viewer->Refresh();
          }
        }
      } else if (auto msg = dynamic_cast<msg::IMU *>(raw_msg)) {
        est->InertialMeas(msg->ts_, msg->gyro_, msg->accel_);
        // if (viewer) {
        //   viewer->Update_gsb(est->gsb());
        //   viewer->Update_gsc(est->gsc());
        // }
      } else {
        LOG(FATAL) << "Invalid entry type.";
      }

      traj_est.emplace_back(est->ts(), est->gsb());
      ostream << StrFormat("%ld", est->ts().count()) << " "
        << est->gsb().translation().transpose() << " "
        << est->gsb().rotation().log().transpose() << std::endl;

      // std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    // Dump output graph
    if (!FLAGS_graphout.empty()) {
      GraphWriter GW;
      GW.CollectGraph(Graph::instance());
      GW.CollectGraph(Mapper::instance());
      GW.WriteDot(FLAGS_graphout);
    }

  } else {
    LOG(FATAL) << "failed to open output file @ " << FLAGS_out;
  }
  // while (viewer) {
  //   viewer->Refresh();
  //   usleep(30);
  // }
}
