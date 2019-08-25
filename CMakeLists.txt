cmake_minimum_required(VERSION 3.5)
project(feh)

option(BUILD_ROSNODE "wrap the system in a rosnode" OFF)
option(BUILD_G2O "build with g2o support" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -Wno-narrowing -Wno-register -fPIC -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# add_definitions(-DOMP_NUM_THREADS=4)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
add_definitions(-DNDEBUG)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# disable logging whose severity level is below the given integer
add_definitions(-DGOOGLE_STRIP_LOG=1)


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
        /usr/lib/x86_64-linux-gnu
)

include_directories(
        ${PROJECT_SOURCE_DIR}/common
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/thirdparty/gflags/include
        ${PROJECT_SOURCE_DIR}/thirdparty/glog/include
        ${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/include
        ${PROJECT_SOURCE_DIR}/thirdparty/jsoncpp/include
        ${PROJECT_SOURCE_DIR}/thirdparty/eigen-3.3.7/include/eigen3

        ${PROJECT_SOURCE_DIR}/thirdparty/pybind11/include
        ${JSONCPP_INCLUDE_DIRS}
        /usr/local/include/opencv2
        /usr/include/suitesparse
        # /opt/ros/kinetic/include/opencv-3.3.1-dev
        # ${PROJECT_SOURCE_DIR}/thirdparty/libigl/include
        # ${PROJECT_SOURCE_DIR}/thirdparty/gperftools/include
)



enable_testing()
add_subdirectory(thirdparty/googletest)
add_subdirectory(thirdparty/gflags)
add_subdirectory(thirdparty/abseil-cpp)
if (BUILD_G2O)
  link_directories(${PROJECT_SOURCE_DIR}/thirdparty/g2o/release/lib)
  include_directories(${PROJECT_SOURCE_DIR}/thirdparty/g2o/release/include)
  list(APPEND deps
    g2o_core
    g2o_solver_dense
    g2o_solver_cholmod
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_types_slam3d
    g2o_types_sba
    g2o_stuff
    cholmod
    cxsparse
    )
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
set(libxivo common xest xapp)
if (BUILD_G2O)
  list(APPEND libxivo xopt)
endif(BUILD_G2O)
target_link_libraries(pyxivo PRIVATE ${libxivo})
