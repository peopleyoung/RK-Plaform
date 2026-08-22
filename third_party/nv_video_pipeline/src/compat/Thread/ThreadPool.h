#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace toolkit {

class ThreadPool {
public:
    enum Priority {
        PRIORITY_HIGHEST
    };

    ThreadPool(size_t thread_count = 1, Priority = PRIORITY_HIGHEST, bool = true) {
        for (size_t index = 0; index < thread_count; ++index) {
            workers_.emplace_back([this]() {
                worker();
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        for (auto& thread : workers_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    template <typename Function>
    void async(Function&& function) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace(std::forward<Function>(function));
        }
        condition_.notify_one();
    }

private:
    void worker() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this]() {
                    return stopping_ || !tasks_.empty();
                });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    bool                              stopping_ = false;
    std::mutex                        mutex_;
    std::condition_variable           condition_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread>          workers_;
};

}  // namespace toolkit
