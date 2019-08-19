#include "camera_factory.h"
#include "json/json.h"
#include "Eigen/Dense"

using namespace feh;
using namespace std;

int main()
{
  ATANCamera<float> *cam = new ATANCamera<float>(480, 640, 400, 400, 320, 240, 1.0);
  Eigen::Vector2f xc(100, 100);
  std::cout << "rows=" << cam->rows() << "; cols=" << cam->cols() << std::endl;
  cam->Project(xc);

  delete cam;
}

