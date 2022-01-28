#include <pnp_python_binding.h>
#include <pnp_ransac.h>
#include <utils/cvl/pose.h>

namespace py = pybind11;
using namespace cvl;

namespace pycvl {



std::vector<Vector3d> np2cvl_vec3d(py::array_t<double> arr){
    auto r = arr.unchecked<2>();  // Checks that input array is of ndim 2

    std::vector<Vector3d> vect_arr; vect_arr.resize(arr.shape(0));
    for(int i=0;i<arr.shape(0);++i)
        for(int j=0;j<arr.shape(1);++j)
            vect_arr[i][j]=arr.at(i,j);
    return vect_arr;
}

std::vector<Vector2d> np2cvl_vec2d(py::array_t<double> arr){
    std::vector<Vector2d> vect_arr; vect_arr.resize(arr.shape(0));
    for(int i=0;i<arr.shape(0);++i)
        for(int j=0;j<arr.shape(1);++j)
            vect_arr[i][j]=arr.at(i,j);
    return vect_arr;
}


//std::vector<double> pnp(std::vector<double> xs_in;std::vector<double> yns){
py::array_t<double> py_pnp(py::array_t<double> xs_in, py::array_t<double> ys_in, double threshold){
    std::vector<Vector3d> xs = np2cvl_vec3d(xs_in);
    std::vector<Vector2d> ys = np2cvl_vec2d(ys_in);

    PoseD res = pnp_ransac(xs, ys, PnpParams(threshold));

    auto pose = res.get4x4();
    std::vector<double> output; output.reserve(16);
    //for(double d:pose)        output.push_back(d);
    for(int row=0;row<4;++row)
        for(int col=0;col<4;++col)
            output.push_back(pose(row,col));
    //auto np_pose = py::array_t<double>(std::vector<ptrdiff_t>{4, 4}, pose.begin());

    /*
    std::vector<Vector3d> xs; xs.reserve(xs_in.size()/3);
    for(int i=0;i<xs_in.size();i+=3)
        xs.push_back(Vector3d(xs_in[i],xs_in[i+1],xs_in[i+2]));
    */


    return py::array_t<double>(std::vector<ptrdiff_t>{4, 4},&output[0]);
}


PYBIND11_MODULE(pnp_python_binding, m){
    m.doc() = "PyBind11 binding for lambda twist-based PnP";

    m.def("pnp", &py_pnp, "Calculates and returns a pose given 2D and 3D point correspondences, OBS! 2D image coordinates should be normalized",
          py::arg("xs_in"), py::arg("ys_in"), py::arg("threshold")=0.001);
}

}
