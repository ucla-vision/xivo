#pragma once
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>



// no namespace for simplifying pybindings

namespace py = pybind11;

namespace pycvl {


 /**   

   The vector returned is the pose i.e. rigid transform represented as a 4x4 matrix [R|t] stored r0,c0.
   The vector xs shall contain Nx3 doubles, these are 3d points, stored x0,y0,z0, x1,y1,z1 by default in numpy
   The vector yns shall contain Nx2 doubles, these are the pinhole normalized coordinates
*/
py::array_t<double> pnp(py::array_t<double> xs_in, py::array_t<double> ys_in, double threshold);

}

