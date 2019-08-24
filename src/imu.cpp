#include "glog/logging.h"

#include "imu.h"

namespace xivo {

IMUState &IMUState::operator+=(const Tangent &dX) {
  int idx{0};
  // accel scaling
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      Ca(i, j) += dX(idx++);
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Cg(i, j) += dX(idx++);
    }
  }
  CHECK(idx == 15);
}

IMU::IMU(const Mat3 &Ca, const Mat3 &Cg) : X_{Ca, Cg} {
  CHECK(Ca(1, 0) == 0 && Ca(2, 0) == 0 && Ca(2, 1) == 0);
}

} // namespace xivo
