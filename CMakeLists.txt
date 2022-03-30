cmake_minimum_required(VERSION 3.5)

# vision lab visual-inertial navigation system
project(vl_vins)

# Set operating system variables
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(IS_LINUX TRUE)
endif ()
if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(IS_MAC TRUE)
endif ()
message(STATUS "IS_LINUX ${IS_LINUX}")
message(STATUS "IS_MAC ${IS_MAC}")

option(BUILD_G2O "build with g2o support" OFF)
option(USE_GPERFTOOLS "use gperf for performance profiling" OFF)

if (USE_GPERFTOOLS)
  add_definitions(-DUSE_GPERFTOOLS)
endif (USE_GPERFTOOLS)

if (BUILD_G2O)
  add_definitions(-DUSE_G2O)
endif (BUILD_G2O)

set(CMAKE_CXX_STANDARD 17)
# TODO (xiaohfei): We should turn on -Wall and -Werror to eliminate all the warnings.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wno-narrowing -Wno-register -fPIC -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")

# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_BUILD_TYPE "Release")
#add_definitions(-DNDEBUG)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# disable logging whose severity level is below the given integer
add_definitions(-DGOOGLE_STRIP_LOG=1)

# add_definitions(-DEIGEN_DEFAULT_TO_ROW_MAJOR)
add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)

set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/thirdparty")
list(APPEND CMAKE_PREFIX_PATH ${THIRD_PARTY_DIR})

find_package(OpenCV REQUIRED PATHS ${THIRD_PARTY_DIR}/opencv/install)
find_package(Python3 REQUIRED Interpreter Development)
find_package(Pangolin REQUIRED)
find_package(jsoncpp REQUIRED)

# Google stuff
find_package(gflags REQUIRED)
find_package(GTest REQUIRED NAMES GTest gtest googletest GoogleTest)
find_package(glog REQUIRED)
find_package(Ceres REQUIRED)

# DBoW2 
find_package(DBoW2 REQUIRED)
# Create an interface target for DBoW2 (since DBoW2 does not export headers & libraries as a target).
# We will add the DBoW2 target to our dependency list in a bit.
add_library(DBoW2 INTERFACE)
target_include_directories(DBoW2 INTERFACE ${DBoW2_INCLUDE_DIR})
target_link_libraries(DBoW2 INTERFACE ${DBoW2_LIBS})

# The dependency PNP does not properly export a target for easy integration,
# so we create an interface target here, and will add the interface target to 
# the dependency list in a bit.
add_library(pnp INTERFACE)
target_include_directories(pnp INTERFACE ${THIRD_PARTY_DIR}/pnp)
target_link_libraries(pnp INTERFACE ${THIRD_PARTY_DIR}/pnp/build/liblibpnp.a)

# TODO (xiaohfei): this is not working since gperftools does not properly generate cmake config files.
# Need to manually add gperftools if needed.
if (USE_GPERFTOOLS)
  find_package(gperftools REQUIRED)
endif()


list(APPEND DEPS
  # OpenCV
  opencv_highgui
  opencv_features2d
  opencv_core
  opencv_video
  opencv_imgproc
  opencv_imgcodecs
  opencv_xfeatures2d

  # Google stuff
  glog::glog
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
  GTest::gmock_main
  gflags_static

  # Ceres::ceres should carry over all its own dependencies, so we don't need 
  # to add the following dependencies: cxsparse, cholmod, lapack, and blas. 
  # But if needed, you can add them back.
  Ceres::ceres

  # Use jsoncpp_static if you are building modules of xivo as static libraryies
  # jsoncpp_static
  jsoncpp_lib

  # Target "pangolin" exported by Pangolin should carry over its own dependencies, i.e., 
  # GLEW and GL (in Linux). If needed, you can manually add them.
  pangolin

  # Loop closure dependencies. 
  pnp
  DBoW2
)

link_libraries(${DEPS})


include_directories(
  ${CMAKE_SOURCE_DIR}/common
  ${CMAKE_SOURCE_DIR}/src
)

enable_testing()
if (BUILD_G2O)
  link_directories(${THIRD_PARTY_DIR}/g2o/release/lib)
  include_directories(${THIRD_PARTY_DIR}/g2o/release/include)
  link_libraries(
    g2o_core
    g2o_solver_dense
    g2o_solver_cholmod
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_types_slam3d
    g2o_types_sba
    g2o_stuff
)
endif(BUILD_G2O)

add_subdirectory(common)
add_subdirectory(src)

########################################
# PYTHON BINDING
########################################
# NOTE: to build with a specific python version
# cmake -DPYTHON_EXECUTABLE=path/to/python ..
# By default, the python binding generated is only compatible with
# your default python interpreter, which you can check by typing
# "which python" in your terminal.
# If you see an error saying "Python.h: No such file or directory", you probably
# need to "sudo apt-get install python3-dev" assuming you are binding to python3.
set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(${THIRD_PARTY_DIR}/pybind11)
pybind11_add_module(pyxivo MODULE pybind11/pyxivo.cpp)
set(pybind_deps common xest xapp)
if (BUILD_G2O)
  list(APPEND pybind_deps xopt)
endif(BUILD_G2O)
target_link_libraries(pyxivo PRIVATE ${pybind_deps})
