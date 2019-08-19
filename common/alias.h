#pragma once

#include "rodrigues.h"
#include "se3.h"

namespace feh {

using ftype = double;

using Mat3 = Eigen::Matrix<ftype, 3, 3>;
using Vec3 = Eigen::Matrix<ftype, 3, 1>;
using Mat4 = Eigen::Matrix<ftype, 4, 4>;
using Vec4 = Eigen::Matrix<ftype, 4, 1>;
using Mat2 = Eigen::Matrix<ftype, 2, 2>;
using Vec2 = Eigen::Matrix<ftype, 2, 1>;
using Vec6 = Eigen::Matrix<ftype, 6, 1>;
using Vec9 = Eigen::Matrix<ftype, 9, 1>;
using Mat23 = Eigen::Matrix<ftype, 2, 3>;
using Mat32 = Eigen::Matrix<ftype, 3, 2>;
using Mat34 = Eigen::Matrix<ftype, 3, 4>;
using Mat93 = Eigen::Matrix<ftype, 9, 3>;
using Mat39 = Eigen::Matrix<ftype, 3, 9>;
using MatX = Eigen::Matrix<ftype, Eigen::Dynamic, Eigen::Dynamic>;
using VecX = Eigen::Matrix<ftype, Eigen::Dynamic, 1>;

using Mat3f = Eigen::Matrix<float, 3, 3>;
using Mat4f = Eigen::Matrix<float, 4, 4>;
using Mat34f = Eigen::Matrix<float, 3, 4>;
using Mat23f = Eigen::Matrix<float, 2, 3>;
using Mat24f = Eigen::Matrix<float, 2, 4>;
using MatXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3>;

using Mat3d = Eigen::Matrix<double, 3, 3>;
using Mat4d = Eigen::Matrix<double, 4, 4>;
using Mat34d = Eigen::Matrix<double, 3, 4>;
using Mat23d = Eigen::Matrix<double, 2, 3>;
using Mat24d = Eigen::Matrix<double, 2, 4>;
using MatXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using MatX3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;

using Vec2f = Eigen::Matrix<float, 2, 1>;
using Vec3f = Eigen::Matrix<float, 3, 1>;
using Vec4f = Eigen::Matrix<float, 4, 1>;
using VecXf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using Vec2d = Eigen::Matrix<double, 2, 1>;
using Vec3d = Eigen::Matrix<double, 3, 1>;
using Vec4d = Eigen::Matrix<double, 4, 1>;
using VecXd = Eigen::Matrix<double, Eigen::Dynamic, 1>;

using Vec2i = Eigen::Matrix<int, 2, 1>;
using Vec3i = Eigen::Matrix<int, 3, 1>;
using Vec4i = Eigen::Matrix<int, 4, 1>;
using VecXi = Eigen::Matrix<int, Eigen::Dynamic, 1>;
using MatXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using MatX3i = Eigen::Matrix<int, Eigen::Dynamic, 3>;

using SE3 = SE3Type<ftype>;
using SO3 = SO3Type<ftype>;

using SE3f = SE3Type<float>;
using SO3f = SO3Type<float>;

using SE3d = SE3Type<double>;
using SO3d = SO3Type<double>;

static const ftype eps = 1e-4f;

} // namespace feh
