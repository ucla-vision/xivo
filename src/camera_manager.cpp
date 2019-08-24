// Singleton camera manager to create and manage different camera models.  
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "camera_manager.h"

namespace xivo {
std::unique_ptr<CameraManager> CameraManager::instance_ = nullptr;

CameraManager *CameraManager::Create(const Json::Value &cfg) {
  if (!instance_) {
    instance_ = std::unique_ptr<CameraManager>(new CameraManager(cfg));
  }
  return instance_.get();
}

CameraManager::CameraManager(const Json::Value &cfg) : model_{Unknown{}} {

  auto cam_model = cfg["model"].asString();
  int rows = cfg["rows"].asInt();
  int cols = cfg["cols"].asInt();
  number_t fx = cfg["fx"].asDouble();
  number_t fy = cfg["fy"].asDouble();
  number_t cx = cfg["cx"].asDouble();
  number_t cy = cfg["cy"].asDouble();

  if (cam_model == "atan") {
    fx = cols * fx;
    fy = rows * fy;
    cx = cols * cx;
    cy = rows * cy;
    number_t w = cfg["w"].asDouble();
    model_ = ATAN{rows, cols, fx, fy, cx, cy, w};
    dim_ = ATAN::DIM;
  } else if (cam_model == "equidistant") {
    auto k0123 = GetVectorFromJson<number_t, 4>(cfg, "k0123");
    int max_iter = cfg.get("max_iter", 15).asInt();

    model_ = EquiDist{rows,     cols,     fx,       fy,       cx,      cy,
                      k0123[0], k0123[1], k0123[2], k0123[3], max_iter};
    dim_ = EquiDist::DIM;
  } else if (cam_model == "radtan") {
    throw std::runtime_error("Radial-Tangential Model NOT implemented");
  } else if (cam_model == "pinhole") {
    model_ = Pinhole{rows, cols, fx, fy, cx, cy};
    dim_ = Pinhole::DIM;
  } else {
    throw std::invalid_argument("invalid camera model");
  }

  rows_ = rows;
  cols_ = cols;
  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;
  fl_ = 0.5 * std::sqrt(fx * fx + fy * fy);
}

} // namespace xivo
