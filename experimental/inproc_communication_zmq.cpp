// Getting familar with ZeroMQ library.
// Author: Xiaohan Fei
// Reference:
// http://zguide.zeromq.org/page:all#sockets-and-patterns
// Example:
// https://ogbe.net/blog/zmq_helloworld.html
// http://zguide.zeromq.org/cpp:mspoller
#include <thread>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

#include "zmq.hpp"
#include "glog/logging.h"

class Node {
public:

  Node(zmq::context_t &context, const std::string &port):
    sub_{context, ZMQ_SUB},
    msg_counter_{0}
  {
    try {
      sub_.connect(port);
      LOG(INFO) << "Connected to port: " << port;
    } catch (const std::exception &e) {
      LOG(FATAL) << "Failed to connect to port: " << port;
    }
    sub_.setsockopt(ZMQ_SUBSCRIBE, "", 0);
    // static_cast required due to zmq bug in C++11
    items_.push_back({static_cast<void *>(sub_), 0, ZMQ_POLLIN, 0});
  }

  void run(int max_msg) {
    while (true) {
      zmq::message_t rx_msg;
      zmq::poll(&items_[0], 1, -1); // http://api.zeromq.org/2-1:zmq-poll
      if (items_[0].revents & ZMQ_POLLIN) {  // revents == received events, make sure the revents is ZMQ_POLLIN (read)
        sub_.recv(&rx_msg);
        std::string rx_str;
        rx_str.assign(static_cast<char *>(rx_msg.data()), rx_msg.size()); // assign ?
        std::cout << "Received: " << rx_str << std::endl;
        if (++msg_counter_ == max_msg) {
          break;
        }
      }
    }
  }

  zmq::socket_t sub_;
  std::vector<zmq::pollitem_t> items_;
  int msg_counter_;
};

int main() 
{
  zmq::context_t context(0);
  zmq::socket_t publisher(context, ZMQ_PUB);
  std::string port("inproc://5566");
  publisher.bind(port);

  Node node(context, port); 

  size_t nmsg = 10; 
  std::thread subs_thread(&Node::run, &node, nmsg);

  for (size_t i = 0; i < nmsg; ++i) {
    std::string content("Hello " + std::to_string(i));
    zmq::message_t msg(content.size());
    std::copy(content.c_str(), content.c_str() + content.size(), static_cast<char*>(msg.data()));
    publisher.send(msg);
  }

  subs_thread.join();
}
