// Elements for pose graph optimization.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <tuple>

#include "g2o_setup.h"

#include "alias.h"
#include "project.h"

namespace xivo {

class GroupVertex: public g2o::BaseVertex<6, SE3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GroupVertex() = default;

  virtual void setToOriginImpl() {
    _estimate = SE3{};  // gsb: body-to-spatial
  }

  virtual void oplusImpl(const double* update) {
    _estimate.R() *= SO3::exp(Eigen::Map<const Vec3>(update));
    _estimate.T() += Eigen::Map<const Vec3>(update+3);
  }

  virtual bool read(std::istream& is) {
    std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
    return false;
  }
  virtual bool write(std::ostream& os) const {
    std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
    return false;
  }
};

class FeatureVertex: public g2o::BaseVertex<3, Vec3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FeatureVertex() = default;

  virtual void setToOriginImpl() {
    _estimate.setZero();
  }

  virtual void oplusImpl(const double* update) {
    for (int i = 0; i < 3; ++i)
      _estimate[i] += update[i];
  }

  virtual bool read(std::istream& is) {
    std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
    return false;
  }
  virtual bool write(std::ostream& os) const {
    std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
    return false;
  }
};

class Edge: public g2o::BaseBinaryEdge<2, Vec2,  FeatureVertex, GroupVertex> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Edge() = default;

  void computeError() {
    const FeatureVertex* fv = static_cast<const FeatureVertex*>(_vertices[0]);
    const GroupVertex* gv = static_cast<const GroupVertex*>(_vertices[1]);
    // gsb.inv -> gbs
    Vec3 Xb = gv->estimate().inv() * fv->estimate();
    _error = Xb.head<2>() / Xb(2) - _measurement;
  }

  void linearizeOplus() override {
    const FeatureVertex* fv = static_cast<const FeatureVertex*>(_vertices[0]);
    const GroupVertex* gv = static_cast<const GroupVertex*>(_vertices[1]);

    const Mat3& Rsb = gv->estimate().R().matrix();
    Mat3 Rsb_t = Rsb.transpose();
    const Vec3& Tsb = gv->estimate().T();

    const Vec3& Xs = fv->estimate();

    // Vec3 Xb = gv->estimate().inv() * fv->estimate();
    Vec3 Xb = Rsb_t * (Xs - Tsb);
    Mat23 derror_dXb;
    project(Xb, &derror_dXb);
    Mat3 dXb_dXs = Rsb_t;
    Mat3 dXb_dWsb = hat(Rsb_t * (Xs - Tsb));
    Mat3 dXb_dTsb = -Rsb_t;
    _jacobianOplusXi = derror_dXb * dXb_dXs;
    _jacobianOplusXj << derror_dXb * dXb_dWsb, derror_dXb * dXb_dTsb;
  }

  virtual bool read(std::istream& is) {
    std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
    return false;
  }

  virtual bool write(std::ostream& os) const {
    std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
    return false;
  }
};

struct FeatureAdapter {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id;
  Vec3 Xs;  // 3D coordinates in spatial frame
};

struct GroupAdapter {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id;
  SE3 gsb;  // body to spatial transformation
};

using ObsAdapterG = std::tuple<GroupAdapter, Vec2, Mat2>;
using VectorObsAdapterG = std::vector<ObsAdapterG>;
using ObsAdapterF = std::tuple<FeatureAdapter, Vec2, Mat2>;
using VectorObsAdapterF = std::vector<ObsAdapterF>;

} // namespace xivo
