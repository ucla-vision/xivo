#pragma once
// stl
#include <memory>
#include <thread>
#include <typeinfo>
// 3rdparty
#include "ProducerConsumerQueue.h"
#include "glog/logging.h"

// own
#include "utils.h"

namespace xivo {

// my little process wrapper of
// the single producer single consumer queue
// from facebook's folly library
template <typename MessageT> class Process {
public:
  // struct Block : public MessageT {
  //   Block() : ready_{false} {}
  //   std::atomic<bool> ready_;
  // };

  Process(uint32_t size = 1000) : worker_{nullptr}, queue_{size} {}

  Process(const Process &) = delete;
  Process &operator=(const Process &) = delete;

  virtual void Wait() {
    while (!queue_.isEmpty()) {
      usleep(1);
    }
  }

  virtual ~Process() {
    if (worker_) {
      worker_->join();
      delete worker_;
    }
    LOG(INFO) << "process stopped";
  }

  void Start() {
    worker_ = new std::thread([this]() {
      for (;;) {
        std::unique_ptr<MessageT> message;
        while (!queue_.read(message)) {
          continue;
        }
        if (!this->Handle(message.get())) {
          LOG(FATAL) << "cannot handle unknown message type";
        }
      }
    });
  }

  void Enqueue(std::unique_ptr<MessageT> message) {
    // DLOG(INFO) << "enqueueing message ..." << std::endl;
    while (!queue_.write(std::move(message))) {
      continue;
    }
    // DLOG(INFO) << "message euqueued" << std::endl;
  }

protected:
  // Message handler. Return true if the message is known to this process and
  // successfully processed; otherwise return false.
  // This is like the "chain of responsibility" design pattern, where the
  // message
  // will be passed to a child class only if the parent class cannot handle it.
  virtual bool Handle(MessageT *message) {
    // LOG(INFO) << "Template process; implement actual handle function here";
    // NOTE: Handle function is in charge of destroying the message if needed.
    // if (auto msg = dynamic_cast<Block*>(message)) {
    //   msg->ready_ = true;
    //   delete message;
    //   return true;
    // }
    return false;
  }

private:
  own<std::thread *> worker_;
  // folly::ProducerConsumerQueue<own<MessageT *>> queue_; // the queue owns the
  // pointers and need to delete them after use
  folly::ProducerConsumerQueue<std::unique_ptr<MessageT>>
      queue_; // the queue owns the pointers and need to delete them after use
};

} // namespace xivo
