// factory method to create a system
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "param.h"
#include "camera_manager.h"
#include "mm.h"
#include "tracker.h"
#include "graph.h"
#include "estimator.h"

#ifdef USE_G2O
#include "optimizer.h"
#endif

namespace xivo {

EstimatorPtr CreateSystem(const Json::Value &cfg) {
  static bool system_created{false};

  if (system_created) {
    return Estimator::instance();
  }

  // Initialize paramter server
  ParameterServer::Create(cfg);
  LOG(INFO) << "Parameter server created";

  // Load camera parameters
  auto cam_cfg = cfg["camera_cfg"].isString()
                     ? LoadJson(cfg["camera_cfg"].asString())
                     : cfg["camera_cfg"];
  Camera::Create(cam_cfg);
  LOG(INFO) << "Camera created";

  // Initialize memory manager
  MemoryManager::Create(cfg["memory"].get("max_features", 256).asInt(),
                        cfg["memory"].get("max_groups", 128).asInt());
  LOG(INFO) << "Memory management unit created";

  // Initialize tracker
  auto tracker_cfg = cfg["tracker_cfg"].isString()
                         ? LoadJson(cfg["tracker_cfg"].asString())
                         : cfg["tracker_cfg"];
  Tracker::Create(tracker_cfg);
  LOG(INFO) << "Tracker created";

  // Initialize the visibility graph
  Graph::Create();
  LOG(INFO) << "Visibility graph created";

#ifdef USE_G2O
  // Initialize the optimizer
  Optimizer::Create(cfg["optimizer"]);
  LOG(INFO) << "Optimizer created";
#endif

  // Initialize the estimator
  Estimator::Create(cfg);
  LOG(INFO) << "Estimator created";

  system_created = true;

  return Estimator::instance();
}

 
} // namespace xivo
