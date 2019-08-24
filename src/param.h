// A wrapper of JsonCpp's Value class
// using as a globally accessible parameter server.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "json/json.h"
#include <memory>

namespace xivo {

class ParameterServer;
using ParameterServerPtr = ParameterServer *;
class ParameterServer : public Json::Value {
public:
  static ParameterServerPtr Create(const Json::Value &value);
  static ParameterServerPtr instance() { return instance_.get(); }

private:
  ParameterServer() = delete;
  ParameterServer(const ParameterServer &) = delete;
  ParameterServer &operator=(const ParameterServer &) = delete;
  ParameterServer(const Json::Value &value);

  static std::unique_ptr<ParameterServer> instance_;
};

} // namespace xivo
