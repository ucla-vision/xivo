cmake_minimum_required(VERSION 3.5)
project(xivo)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(roscpp REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  std_msgs
  image_transport
  message_generation
  sensor_msgs
  cv_bridge
  tf
  nav_msgs
  roscpp
  pcl_ros
  geometry_msgs
)


add_message_files(
  FILES
  FeatureData.msg
  FeatureMap.msg
  FullState.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)


# Declare the message files to be built
catkin_package(CATKIN_DEPENDS
  dynamic_reconfigure
  std_msgs
  message_runtime
  image_transport
  dynamic_reconfigure
  sensor_msgs
  tf
  cv_bridge
  nav_msgs
  geometry_msgs
  roscpp
  CATKIN_DEPENDS
  INCLUDE_DIRS
  LIBRARIES
)

include_directories(
  ${CMAKE_SOURCE_DIR}/common
  ${CMAKE_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/msg/cpp/
  ${catkin_INCLUDE_DIRS}
)

link_directories(
  ${CMAKE_SOURCE_DIR}/lib
  /opt/ros/kinetic/lib/x86_64-linux-gnu
)

set(libxivo xest xapp)
if (BUILD_G2O)
  list(APPEND libxivo xopt)
endif(BUILD_G2O)


add_executable(mynode main.cpp simple_node.cpp simple_node.h)
target_link_libraries(mynode
  ${libxivo}
  gflags::gflags
  glog
  jsoncpp
  opencv_core
  opencv_imgcodecs
  opencv_highgui
  ${catkin_LIBRARIES}
  ${roscpp_LIBRARIES})
