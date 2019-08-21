cmake_minimum_required(VERSION 3.5)
project(feh)

option(BUILD_ROSNODE "wrap the system in a rosnode" OFF)
option(BUILD_G2O "build with g2o support" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -Wno-narrowing -Wno-register -g -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")
set(CMAKE_BUILD_TYPE "Release")
add_definitions(-DNDEBUG)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# add_definitions(-DEIGEN_DEFAULT_TO_ROW_MAJOR)
add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)

# comment out the following line to disable NaN checks on Jacobians
find_package(OpenCV REQUIRED)

link_directories(
        ${PROJECT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/gflags/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/glog/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/jsoncpp/lib
        # ${PROJECT_SOURCE_DIR}/thirdparty/gperftools/lib
        /usr/local/lib
        # /opt/ros/kinetic/lib/x86_64-linux-gnu
        # /usr/lib
        # /usr/lib/x86_64-linux-gnu
)

include_directories(
        ${PROJECT_SOURCE_DIR}/common
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/thirdparty/gflags/include
        ${PROJECT_SOURCE_DIR}/thirdparty/glog/include
        ${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/include
        ${PROJECT_SOURCE_DIR}/thirdparty/jsoncpp/include
        ${PROJECT_SOURCE_DIR}/thirdparty/eigen3/include/eigen3
        ${PROJECT_SOURCE_DIR}/thirdparty/pybind11/include
        ${JSONCPP_INCLUDE_DIRS}
        /usr/local/include/opencv2
        # /opt/ros/kinetic/include/opencv-3.3.1-dev
        # ${PROJECT_SOURCE_DIR}/thirdparty/libigl/include
        # ${PROJECT_SOURCE_DIR}/thirdparty/gperftools/include
)



enable_testing()
add_subdirectory(thirdparty/googletest)
add_subdirectory(thirdparty/gflags)
add_subdirectory(thirdparty/abseil-cpp)
if (BUILD_G2O)
  add_subdirectory(thirdparty/g2o)
endif(BUILD_G2O)

# feh
add_subdirectory(common)
add_subdirectory(src)
if (BUILD_ROSNODE)
  add_subdirectory(node)
endif(BUILD_ROSNODE)

########################################
# PYTHON BINDING
########################################
# NOTE: to build with a specific python version
# cmake -DPYTHON_EXECUTABLE=path/to/python ..
# By default, the python binding generated is only compatible with
# your default python interpreter, which you can check by typing
# "which python" in your terminal.
set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(thirdparty/pybind11)
pybind11_add_module(pyxivo MODULE pybind11/pyxivo.cpp)
target_link_libraries(pyxivo PRIVATE common estimator application)
