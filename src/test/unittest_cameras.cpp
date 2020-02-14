#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>

#include "core.h"

#include <random>

using namespace Eigen;
using namespace xivo;


TEST(Cameras, Pinhole) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["perfect_pinhole"]);

  std::default_random_engine generator;
  std::uniform_real_distribution<number_t> distribution(0.0, 5.0);

  Vec2 px;
  px(0) = distribution(generator);
  px(1) = distribution(generator);

  Vec2 px_projected = cam->Project(px);
  Vec2 px2 = cam->UnProject(px_projected);

  EXPECT_FLOAT_EQ(px(0), px2(0));
  EXPECT_FLOAT_EQ(px(1), px2(1));
}



TEST(Cameras, RadTanNewton) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["realsense_radtan"]);

  std::default_random_engine generator;
  std::uniform_real_distribution<number_t> distribution(0.0, 5.0);

  Vec2 px;
  px(0) = distribution(generator);
  px(1) = distribution(generator);

  Vec2 px_projected = cam->Project(px);
  Vec2 px2 = cam->UnProject(px_projected);

  EXPECT_FLOAT_EQ(px(0), px2(0));
  EXPECT_FLOAT_EQ(px(1), px2(1));
}