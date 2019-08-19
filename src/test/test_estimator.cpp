#include <iostream>
#include "gtest/gtest.h"
#include "estimator.h"

using namespace feh;

class EstimatorTest: public ::testing::Test {
protected:
    EstimatorTest() : est_(){
      est_ = std::make_shared<Estimator>(std::string{"../cfg/estimator.json"});
    }
    std::shared_ptr<Estimator> est_;
};

TEST_F(EstimatorTest, Init) {
  std::cout << est_->X_.Rsb.matrix() << std::endl;
}

TEST_F(EstimatorTest, Inertial) {
  for (int i = 0; i < 100; ++i) {
    Vec3 static_acc = 0.1 * RandomVector<3>();
    static_acc += Vec3{0.0, 0.0, 9.8};
    ftype ts = i*0.01;
    est_->InertialMeas(ts, static_acc, Vec3::Random());
    if (est_->gravity_initialized_) {
      // std::cout << err_.segment<3>(Index::T).transpose() << std::endl;
      ASSERT_EQ(est_->err_.head<kMotionSize>().norm(), 0) << "With no correction, error state should be zero";
      est_->vision_initialized_ = true;
      est_->curr_time_ = ts*1.5;
    }

  }
}

