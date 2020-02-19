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


TEST(CamerasEqui, EquiProjectionJacc) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["phab_equi"]);

  std::default_random_engine generator;
  std::uniform_real_distribution<number_t> distribution(0.0, 5.0);

  number_t delta = 1e-6;

  Vec2 px;
  Vec2 px_proj;
  Eigen::Matrix<number_t, 2, Eigen::Dynamic> px_jacc;
  px(0) = distribution(generator);
  px(1) = distribution(generator);

  px_proj = cam->Project(px, nullptr, &px_jacc);

  Vec9 Intrinsics = cam->GetIntrinsics();

  Vec8 dX_fx;
  dX_fx << delta, 0, 0, 0, 0, 0, 0, 0;
  cam->UpdateState(dX_fx);
  Vec2 px_proj_fx = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_fx(0) - px_proj(0)) / delta, px_jacc(0,0));
  EXPECT_FLOAT_EQ((px_proj_fx(1) - px_proj(1)) / delta, px_jacc(1,0));
  cam->UpdateState(-dX_fx);

  Vec8 dX_fy;
  dX_fy << 0, delta, 0, 0, 0, 0, 0, 0;
  cam->UpdateState(dX_fy);
  Vec2 px_proj_fy = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_fy(0) - px_proj(0)) / delta, px_jacc(0,1));
  EXPECT_FLOAT_EQ((px_proj_fy(1) - px_proj(1)) / delta, px_jacc(1,1));
  cam->UpdateState(-dX_fy);

  Vec8 dX_cx;
  dX_cx << 0, 0, delta, 0, 0, 0, 0, 0;
  cam->UpdateState(dX_cx);
  Vec2 px_proj_cx = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_cx(0) - px_proj(0)) / delta, px_jacc(0,2));
  EXPECT_FLOAT_EQ((px_proj_cx(1) - px_proj(1)) / delta, px_jacc(1,2));
  cam->UpdateState(-dX_cx);

  Vec8 dX_cy;
  dX_cy << 0, 0, 0, delta, 0, 0, 0, 0;
  cam->UpdateState(dX_cy);
  Vec2 px_proj_cy = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_cy(0) - px_proj(0)) / delta, px_jacc(0,3));
  EXPECT_FLOAT_EQ((px_proj_cy(1) - px_proj(1)) / delta, px_jacc(1,3));
  cam->UpdateState(-dX_cy);

  Vec8 dX_k0;
  dX_k0 << 0, 0, 0, 0, delta, 0, 0, 0;
  cam->UpdateState(dX_k0);
  Vec2 px_proj_k0 = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_k0(0) - px_proj(0)) / delta, px_jacc(0,4));
  EXPECT_FLOAT_EQ((px_proj_k0(1) - px_proj(1)) / delta, px_jacc(1,4));
  cam->UpdateState(-dX_k0);

  Vec8 dX_k1;
  dX_k1 << 0, 0, 0, 0, 0, delta, 0, 0;
  cam->UpdateState(dX_k1);
  Vec2 px_proj_k1 = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_k1(0) - px_proj(0)) / delta, px_jacc(0,5));
  EXPECT_FLOAT_EQ((px_proj_k1(1) - px_proj(1)) / delta, px_jacc(1,5));
  cam->UpdateState(-dX_k1);

  Vec8 dX_k2;
  dX_k2 << 0, 0, 0, 0, 0, 0, delta, 0;
  cam->UpdateState(dX_k2);
  Vec2 px_proj_k2 = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_k2(0) - px_proj(0)) / delta, px_jacc(0,6));
  EXPECT_FLOAT_EQ((px_proj_k2(1) - px_proj(1)) / delta, px_jacc(1,6));
  cam->UpdateState(-dX_k2);

  Vec8 dX_k3;
  dX_k3 << 0, 0, 0, 0, 0, 0, 0, delta;
  cam->UpdateState(dX_k3);
  Vec2 px_proj_k3 = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_k3(0) - px_proj(0)) / delta, px_jacc(0,7));
  EXPECT_FLOAT_EQ((px_proj_k3(1) - px_proj(1)) / delta, px_jacc(1,7));
  cam->UpdateState(-dX_k3);
}