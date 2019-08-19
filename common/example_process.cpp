#include "process.h"
#include <iostream>

using namespace feh;

struct ExampleMessage {
  virtual ~ExampleMessage() {}
};

struct PrintMessage : public ExampleMessage {
  PrintMessage(int value) : value_{value} {}
  int value_;
};

class ExampleProcess : public Process<ExampleMessage> {
public:
  using MessageT = ExampleMessage;
  ExampleProcess(uint32_t size = 1000) : Process{size}, accumulator_{0} {}

  // Decalre process-specific member variables.
  int accumulator_;

protected:
  void Handle(ExampleMessage *message) {
    // Define the process-specific handle function.
    if (typeid(*message) == typeid(PrintMessage)) {
      accumulator_ += dynamic_cast<PrintMessage *>(message)->value_;
      delete message;
    }
  }
};

int main() {
  ExampleProcess proc{2000};
  proc.Start();
  int accumulator{0};
  for (int i = 0; i < 2000; ++i) {
    usleep(10); // sleep for the process to catch up
    proc.Enqueue(new PrintMessage(i));
    accumulator += i;
  }
  proc.Enqueue(nullptr);
  std::cout << "process.accumulator=" << proc.accumulator_ << std::endl;
  std::cout << "main thread accumulator=" << accumulator << std::endl;
  CHECK_EQ(proc.accumulator_, accumulator)
      << "main thread and process accumulated different values";
}
