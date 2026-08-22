#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>

namespace toolkit {

class Timer {
public:
    using Ptr = std::shared_ptr<Timer>;

    template <typename Poller>
    Timer(float seconds, std::function<bool()> callback, Poller)
        : interval_(std::chrono::milliseconds(static_cast<int64_t>(seconds * 1000))),
          callback_(std::move(callback)),
          worker_([this]() {
              run();
          }) {
    }

    ~Timer() {
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    void run() {
        while (running_) {
            const auto deadline = std::chrono::steady_clock::now() + interval_;
            while (running_ && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (running_ && callback_ && !callback_()) {
                running_ = false;
            }
        }
    }

    std::chrono::milliseconds interval_;
    std::function<bool()>     callback_;
    std::atomic_bool          running_{true};
    std::thread               worker_;
};

}  // namespace toolkit
