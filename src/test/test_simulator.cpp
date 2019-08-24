#include <iostream>
#include "gtest/gtest.h"
#include "simulator.h"

using namespace xivo;

class SimulatorTest: public ::testing::Test {
protected:
    SimulatorTest() : sim_(){
      sim_ = std::make_shared<Simulator>("../cfg/simulator.json");
    }
    SimulatorPtr sim_;
};

TEST_F(SimulatorTest, Init) {
  std::cout << sim_->X_.Rsb.matrix() << std::endl;
}

// TEST_F(SimulatorTest, Traj) {
//     for (auto p : sim_->translations_) {
//         std::cout << p.transpose() << std::endl;
//     }
// }

TEST_F(SimulatorTest, Meas) {
  msg::IMU imu_msg;
  msg::Track track_msg;
  while (sim_->GetMeas(imu_msg, track_msg)) {
    std::cout << sim_->X_.Tsb.transpose() << std::endl;
  }
}


