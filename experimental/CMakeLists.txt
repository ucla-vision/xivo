# TODO (xfei): fill in details
################################################################################
# EXPERIMENTAL
################################################################################

# add_executable(inproc_zmq experimental/inproc_communication_zmq.cpp)
# target_link_libraries(inproc_zmq zmq pthread glog)
#
# add_executable(async_dataloader experimental/async_dataloader.cpp)
# target_link_libraries(async_dataloader zmq pthread glog msckf)

add_executable(test_camera_factory camera_factory.h test_camera_factory.cpp)


