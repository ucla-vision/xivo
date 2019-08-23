#include "optimizer.h"

using namespace feh;
using namespace std;
using namespace Eigen;

class Sample {
public:
  static int uniform(int from, int to);
  static double uniform();
  static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr){
  return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma){
  double x, y, r2;
  do {
    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);
  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to){
  return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform(){
  return uniform_rand(0., 1.);
}

double Sample::gaussian(double sigma){
  return gauss_rand(0., sigma);
}

int main() {
  double PIXEL_NOISE{0.1};
  Json::Value cfg{};
  auto optimizer = Optimizer::Create(cfg);

  vector<Vec3> pts;
  for (int i=0;i<500; ++i) {
    pts.push_back(Vec3{(Sample::uniform()-0.5)*3,
          Sample::uniform()-0.5,
          Sample::uniform()+3});
  }

  vector<SE3> gsb;
  for (int i=0; i<15; ++i) {
    Vec3 trans(i*0.04-1.,0,0);
    SE3 pose{SO3{}, trans};
    gsb.push_back(pose);
  }

  for (int i = 0; i < pts.size(); ++i) {
    const Vec3& pt{pts[i]};
    Vec3 noisy_pt = pt + Vec3{Sample::gaussian(1),
      Sample::gaussian(1),
      Sample::gaussian(1)};

    vector<ObsAdapterG> obs;
    for (int j = 0; j < gsb.size(); ++j) {
      Vec3 Xb = gsb[j].inv() * pt;
      Vec2 xp = Xb.head<2>() / Xb(2);
      Vec2 noisy_xp = xp + Vec2{Sample::gaussian(PIXEL_NOISE),
        Sample::gaussian(PIXEL_NOISE)};
      obs.push_back(std::make_tuple(
            GroupAdapter{j, gsb[j]},
            noisy_xp, Mat2::Identity()));
    }
    optimizer->AddFeature(FeatureAdapter{gsb.size() + i, noisy_pt}, obs);
  }


}
