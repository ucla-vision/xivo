// Asynchronous dataloader.
// Data acquisition (loader or multiple sensors) lives in its own node (might be multiple nodes), and dispatches messages to the estimator node.  This file implements a dataloader node.
// Author: Xiaohan Fei
#include <thread>

#include "zmq.hpp"
#include "glog/logging.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "tumvi.h"
#include "helpers.h"

static std::string tumvi_root = "/home/feixh/Data/tumvi/exported/euroc/512_16/";
static std::string euroc_root = "/home/feixh/Data/euroc/";


namespace feh {
namespace node {

// packet for communication
struct Packet {
  Packet(): type_{UNINITIALIZED}, image_ptr_{nullptr} {}

  Packet(uint64_t ts, 
      ftype rx, ftype ry, ftype rz,
      ftype tx, ftype ty, ftype tz): 
    type_{IMU}, ts_{ts},
    rx_{rx}, ry_{ry}, rz_{rz},
    tx_{tx}, ty_{ty}, tz_{tz},
    image_ptr_{nullptr} 
  {}

  Packet(uint64_t ts, 
      cv::Mat *ptr):
    type_{IMAGE_PTR}, ts_{ts},
    image_ptr_{ptr} 
  {}

  enum: uint8_t {
    UNINITIALIZED = 0,
    IMAGE_PTR = 1,
    IMU = 2
  } type_;

  uint64_t ts_;
  ftype rx_, ry_, rz_, tx_, ty_, tz_;
  cv::Mat *image_ptr_;
};


class DataServer {
public:
  DataServer(zmq::context_t &context, const std::string &port):
    pub_{context, ZMQ_PUB},
    loader_{nullptr}
  {
    try {
      pub_.bind(port);
      LOG(INFO) << "Bound to port: " << port;
    } catch (const std::exception &e) {
      LOG(FATAL) << "Failed to bind to port: " << port;
    }
  }

  void Initialize(const std::string dataset, 
      const std::string root, 
      const std::string seq, 
      int cam_id) {
    std::string image_dir, imu_dir, gt_dir;
    std::tie(image_dir, imu_dir, gt_dir) = GetDirs(dataset, root, seq, cam_id);
    loader_ = std::unique_ptr<TUMVILoader>(new TUMVILoader{image_dir, imu_dir});
    LOG(INFO) << "#entries=" << loader_->size() << std::endl;
  }

  void run() {
    if (!loader_) {
      LOG(FATAL) << "Empty loader!";
    }

    for (int i = 0; i < loader_->size(); ++i) {
      msg::Generic entry = loader_->Get(i);

      // serialize and send the actual message
      Packet packet;
      if (entry.type_ == msg::Generic::IMAGE) {
        cv::Mat *image_ptr = new cv::Mat(cv::imread(entry.image_path_));
         packet = std::move(Packet{entry.ts_, image_ptr});
      } else if (entry.type_ == msg::Generic::IMU) {
         packet = std::move(Packet{entry.ts_, 
          entry.gyro_[0], entry.gyro_[1], entry.gyro_[2],
          entry.accel_[0], entry.accel_[1], entry.accel_[2]});
      }

      zmq::message_t msg(sizeof(packet));
      memcpy(msg.data(), &packet, sizeof(packet));
      pub_.send(msg);
      LOG(INFO) << "Sent " << i+1 << " messages";
    }
  }

private:
  zmq::socket_t pub_;
  std::unique_ptr<TUMVILoader> loader_;
};

class MessageEchoClient {
public:

  MessageEchoClient(zmq::context_t &context, const std::string &port):
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

  void run() {
    while (true) {
      zmq::message_t rx_msg;
      zmq::poll(&items_[0], 1, -1); // http://api.zeromq.org/2-1:zmq-poll
      if (items_[0].revents & ZMQ_POLLIN) {  // revents == received events, make sure the revents is ZMQ_POLLIN (read)
        sub_.recv(&rx_msg);

        // receive message
        Packet packet;
        memcpy(&packet, rx_msg.data(), rx_msg.size());
        if (packet.type_ == Packet::IMU) {
          std::cout << packet.ts_ << " " << packet.rx_ << " " << packet.ry_ << " " << packet.rz_ << " " 
            << packet.tx_ << " " << packet.ty_ << " " << packet.tz_ << std::endl;
        } else if (packet.type_ == Packet::IMAGE_PTR) {
          if (packet.image_ptr_) {
            cv::Mat image{packet.image_ptr_->clone()};
            delete packet.image_ptr_;

            cv::imshow("disp", image);
            char ckey = cv::waitKey(10);
          } else {
            LOG(FATAL) << "Empty image pointer";
          }
        } else {
          LOG(FATAL) << "UN-INITIALIZED PACKET";
        }

        ++msg_counter_;
        LOG(INFO) << "Received " << msg_counter_ << " messages";
      }
    }
  }

  zmq::socket_t sub_;
  std::vector<zmq::pollitem_t> items_;
  int msg_counter_;
  // TODO (xfei): implement a thread-safe message buffer
};

} // namespace node

} // namespace feh


// some constants
static const std::string dataset{"tumvi"};
static const std::string root{"/home/feixh/Data/tumvi/exported/euroc/512_16/"};
static const std::string seq{"room1"};
static const int cam_id{0};

int main() 
{
  zmq::context_t context(0);
  std::string port("inproc://7007");

  // NOTE: launch subscriber first
  feh::node::MessageEchoClient client(context, port);
  std::thread sub_thread(&feh::node::MessageEchoClient::run, &client);

  feh::node::DataServer loader(context, port); 
  loader.Initialize(dataset, root, seq, cam_id);
  std::thread pub_thread(&feh::node::DataServer::run, &loader);

  // join
  pub_thread.join();
  sub_thread.join();
  


}
