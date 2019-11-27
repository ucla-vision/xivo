#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "estimator.h"
#include "opencv2/highgui/highgui.hpp"
#include "utils.h"

// for visualization
#include "viewer.h"
#include "visualize.h"

namespace py = pybind11;
using namespace xivo;

class EstimatorWrapper {
public:
  EstimatorWrapper(const std::string &cfg_path,
                   const std::string &viewer_cfg_path,
                   const std::string &name)
    : name_{name}, imu_calls_{0}, visual_calls_{0} {

    if (!glog_init_) {
      google::InitGoogleLogging("pyxivo");
      glog_init_ = true;
    }

    auto cfg = LoadJson(cfg_path);
    // estimator_ = std::unique_ptr<Estimator>(new Estimator{cfg});
    estimator_ = CreateSystem(cfg);

    if (!viewer_cfg_path.empty()) {
      auto viewer_cfg = LoadJson(viewer_cfg_path);
      viewer_ = std::unique_ptr<Viewer>(new Viewer{viewer_cfg, name});
    }
  }

  void InertialMeas(uint64_t ts, double wx, double wy, double wz, double ax,
                    double ay, double az) {

    // std::cout << "InertialMeas called on " << name_ << " " 
    //   << ++imu_calls_ << " times" << std::endl;

    estimator_->InertialMeas(timestamp_t{ts}, {wx, wy, wz}, {ax, ay, az});

    if (viewer_) {
      viewer_->Update_gsb(estimator_->gsb());
      viewer_->Update_gsc(estimator_->gsc());
    }
  }

  void VisualMeas(uint64_t ts, std::string &image_path) {

    // std::cout << "VisualMeas called on " << name_ << " " 
    //   << ++visual_calls_ << " times" << std::endl;

    auto image = cv::imread(image_path);

    estimator_->VisualMeas(timestamp_t{ts}, image);

    if (viewer_) {
      auto disp = Canvas::instance()->display();

      if (!disp.empty()) {
        LOG(INFO) << "Display image is ready";
        // cv::imshow("tracker", disp);
        // if (cv::waitKey(cfg.get("wait_time", 5).asInt()) == 'q') break;
        viewer_->Update(disp);
      }
    }
  }

  Eigen::Matrix<double, 3, 4> gsb() { return estimator_->gsb().matrix3x4(); }
  Eigen::Matrix<double, 3, 4> gsc() { return estimator_->gsc().matrix3x4(); }

  uint64_t now() const { return estimator_->ts().count(); }

  void Visualize() {
    if (viewer_)
      viewer_->Refresh();
  }

private:
  // std::unique_ptr<Estimator> estimator_;
  EstimatorPtr estimator_;
  std::unique_ptr<Viewer> viewer_;
  static bool glog_init_;
  std::string name_;
  int imu_calls_, visual_calls_;
};

bool EstimatorWrapper::glog_init_{false};

PYBIND11_MODULE(pyxivo, m) {
  m.doc() = "python binding of XIVO (Xiaohan's Inertial-aided Visual Odometry)";
  py::class_<EstimatorWrapper>(m, "Estimator")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &>())
      .def("InertialMeas", &EstimatorWrapper::InertialMeas)
      .def("VisualMeas", &EstimatorWrapper::VisualMeas)
      .def("gsb", &EstimatorWrapper::gsb)
      .def("gsc", &EstimatorWrapper::gsc)
      .def("now", &EstimatorWrapper::now)
      .def("Visualize", &EstimatorWrapper::Visualize);
}
