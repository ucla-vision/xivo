#include "glog/logging.h"

#include "geometry.h"
#include "metrics.h"

namespace feh {

static std::tuple<ftype, SE3>
AbsoluteTrajectoryError(const std::vector<Vec3> &Y,
                        const std::vector<Vec3> &X) {
  auto gYX = TrajectoryAlignment(Y, X);

  LOG(INFO) << "computing ATE";
  ftype res{0};
  for (int i = 0; i < X.size(); ++i) {
    // AR=RB
    Vec3 r = Y[i] - gYX * X[i];
    res += r.squaredNorm();
  }
  res = sqrt(res / X.size());
  LOG(INFO) << "ATE=" << res << " meters" << std::endl;
  return std::make_tuple(res, gYX);
}

std::tuple<ftype, SE3> ComputeATE(const std::vector<msg::Pose> &est,
                                  const std::vector<msg::Pose> &gt, ftype res) {
  using std::chrono::abs;

  auto it_est = est.begin();
  auto it_gt = gt.begin();
  timestamp_t bound(uint64_t{res * 1e9}); // seconds -> nanoseconds

  std::vector<Vec3> X, Y;
  while (it_est < est.end() && next(it_gt) < gt.end()) {
    if (it_est->ts_ >= it_gt->ts_ && it_est->ts_ < next(it_gt)->ts_) {
      // make sure the timestamps are close enough
      if (abs(it_est->ts_ - it_gt->ts_) < bound) {
        Y.push_back(it_est->g_.translation());
        X.push_back(it_gt->g_.translation());
      }
      // proceed
      it_est++;
      it_gt++;
    } else if (it_est->ts_ < it_gt->ts_) {
      ++it_est;
    } else {
      ++it_gt;
    }
  }
  auto packet = X.empty() ? std::make_tuple(ftype(-1), SE3{})
                          : AbsoluteTrajectoryError(Y, X);
  return packet;
}

std::tuple<ftype, ftype> ComputeRPE(const std::vector<msg::Pose> &est,
                                    const std::vector<msg::Pose> &gt, ftype dt,
                                    ftype res) {
  using std::chrono::abs;

  auto it_est = est.begin();
  auto it_gt = gt.begin();

  // positional and rotational RPE
  ftype rpe_pos{0}, rpe_rot{0};
  int counter{0};
  while (it_est < est.end() && next(it_gt) < gt.end()) {
    if (it_est->ts_ >= it_gt->ts_ && it_est->ts_ < next(it_gt)->ts_) {
      auto gY = (it_est++)->g_;
      auto gX = (it_gt++)->g_;
      // pair these poses with poses dt seconds apart in time
      SE3 gY2, gX2;
      bool found{false};
      // seconds -> nanoseconds
      timestamp_t desire = it_est->ts_ + timestamp_t{uint64_t(dt * 1e9)};
      timestamp_t bound{uint64_t(res * 1e9)};

      std::vector<msg::Pose>::const_iterator it;
      for (it = it_est; it < est.end(); ++it) {
        if (it->ts_ > desire - bound && it->ts_ < desire + bound) {
          if (abs(it->ts_ - desire) < bound) {
            gY2 = it->g_;
            bound = abs(it->ts_ - desire);
          }
          found = true;
        } else if (it->ts_ > desire + bound) {
          break;
        }
      }
      if (!found)
        continue;

      found = false;
      desire = it_gt->ts_ + timestamp_t{uint64_t(dt * 1e9)};
      bound = timestamp_t{uint64_t(res * 1e9)};

      for (it = it_gt; it < gt.end(); ++it) {
        if (it->ts_ > desire - bound && it->ts_ < desire + bound) {
          if (abs(it->ts_ - desire) < bound) {
            gX2 = it->g_;
            bound = abs(it->ts_ - desire);
          }
          found = true;
        } else if (it->ts_ > desire + bound) {
          break;
        }
      }
      if (!found)
        continue;

      auto dgX = gX.inv() * gX2;
      auto dgY = gY.inv() * gY2;
      rpe_pos += (dgX.inv() * dgY).translation().squaredNorm();
      rpe_rot += (dgX.inv() * dgY).so3().log().squaredNorm();
      ++counter;
    } else if (it_est->ts_ < it_gt->ts_) {
      ++it_est;
    } else {
      ++it_gt;
    }
  }
  LOG(INFO) << "Effective pairs in computing RPE=" << counter;
  if (counter) {
    rpe_pos = sqrt(rpe_pos / counter);
    rpe_rot = sqrt(rpe_rot / counter);
  } else {
    rpe_pos = -1;
    rpe_rot = -1;
  }
  return std::make_tuple(rpe_pos, rpe_rot);
}

} // namespace feh
