// A wrapper of JsonCpp's Value class
// using as a globally accessible parameter server.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "param.h"
#include "glog/logging.h"

namespace feh {

std::unique_ptr<ParameterServer> ParameterServer::instance_ = nullptr;

ParameterServerPtr ParameterServer::Create(const Json::Value &value) {
  if (instance_) {
    LOG(WARNING) << "parameter server already created!";
  } else {
    instance_ = std::unique_ptr<ParameterServer>(new ParameterServer(value));
  }
  return instance_.get();
}

ParameterServer::ParameterServer(const Json::Value &value)
    : Json::Value{value} {}

} // namespace feh
