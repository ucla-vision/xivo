#include <gtest/gtest.h>

#include "core.h"

#include <random>

using namespace Eigen;
using namespace xivo;



TEST(CamerasEqui, EquiProjectUnproject) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["phab_equi"]);

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


TEST(CamerasEqui, EquiProjectionJac) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["phab_equi"]);

  std::default_random_engine generator;
  std::uniform_real_distribution<number_t> distribution(0.0, 5.0);

  number_t delta = 1e-8;

  Vec2 px;
  Vec2 px_projected;
  Mat2 px_jac;
  px(0) = distribution(generator);
  px(1) = distribution(generator);

  px_projected = cam->Project(px, &px_jac);

  Vec2 px1 = px;
  px1(0) += delta;
  Vec2 px1_projected = cam->Project(px1);

  Vec2 px2 = px;
  px2(1) += delta;
  Vec2 px2_projected = cam->Project(px2);

  Vec2 approx_dx1 = (px1_projected - px_projected) / delta;
  Vec2 approx_dx2 = (px2_projected - px_projected) / delta;

  EXPECT_FLOAT_EQ(approx_dx1(0), px_jac(0,0));
  EXPECT_FLOAT_EQ(approx_dx1(1), px_jac(1,0));
  EXPECT_FLOAT_EQ(approx_dx2(0), px_jac(0,1));
  EXPECT_FLOAT_EQ(approx_dx2(1), px_jac(1,1));
}