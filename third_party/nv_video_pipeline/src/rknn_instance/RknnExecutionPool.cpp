#include "RknnExecutionPool.h"

#include <algorithm>
#include <utility>

namespace infer {

RknnExecutionPool::~RknnExecutionPool() {
    stop();
}

bool RknnExecutionPool::init(std::vector<rknn_context> contexts, size_t worker_count, size_t queue_capacity,
                             Destroy destroy) {
    auto cleanup = [&]() {
        if (!destroy) {
            return;
        }
        for (const rknn_context context : contexts) {
            if (context != 0) {
                try {
                    destroy(context);
                } catch (...) {
                }
            }
        }
        contexts.clear();
    };
    if (contexts.empty() || worker_count == 0 || worker_count > contexts.size() || queue_capacity == 0 || !destroy) {
        cleanup();
        return false;
    }
    if (std::any_of(contexts.begin(), contexts.end(), [](rknn_context context) {
            return context == 0;
        })) {
        cleanup();
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!contexts_.empty() || running_) {
        cleanup();
        return false;
    }
    contexts_                = std::move(contexts);
    available_contexts_      = std::deque<rknn_context>(contexts_.begin(), contexts_.end());
    configured_worker_count_ = worker_count;
    queue_capacity_          = queue_capacity;
    destroy_                 = std::move(destroy);
    return true;
}

bool RknnExecutionPool::commit(Task task, Cancel cancel) {
    if (!task || !cancel) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!accepting_ || jobs_.size() >= queue_capacity_) {
        return false;
    }
    jobs_.push_back(Work{std::move(task), std::move(cancel)});
    condition_.notify_one();
    return true;
}

void RknnExecutionPool::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_ || contexts_.empty()) {
        return;
    }
    running_   = true;
    accepting_ = true;
    workers_.reserve(configured_worker_count_);
    for (size_t index = 0; index < configured_worker_count_; ++index) {
        workers_.emplace_back(&RknnExecutionPool::worker, this);
    }
}

void RknnExecutionPool::stop() {
    std::deque<Work> abandoned;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        accepting_ = false;
        running_   = false;
        abandoned.swap(jobs_);
    }
    condition_.notify_all();
    for (const auto& work : abandoned) {
        cancel_safely(work.cancel);
    }
    for (auto& thread : workers_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    workers_.clear();

    std::vector<rknn_context> contexts;
    Destroy                   destroy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        contexts.swap(contexts_);
        available_contexts_.clear();
        configured_worker_count_ = 0;
        queue_capacity_          = 0;
        destroy                  = std::move(destroy_);
    }
    if (destroy) {
        for (rknn_context context : contexts) {
            try {
                destroy(context);
            } catch (...) {
            }
        }
    }
}

size_t RknnExecutionPool::context_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return contexts_.size();
}

size_t RknnExecutionPool::worker_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return configured_worker_count_;
}

bool RknnExecutionPool::create_contexts(size_t context_count, Create create, Duplicate duplicate,
                                        Configure configure, Destroy destroy,
                                        std::vector<rknn_context>& contexts) {
    contexts.clear();
    if (context_count == 0 || !create || !duplicate || !configure || !destroy) {
        return false;
    }
    auto cleanup = [&]() {
        for (rknn_context context : contexts) {
            if (context != 0) {
                destroy(context);
            }
        }
        contexts.clear();
    };

    rknn_context primary = 0;
    if (create(primary) != RKNN_SUCC || primary == 0) {
        if (primary != 0) {
            destroy(primary);
        }
        return false;
    }
    contexts.push_back(primary);
    if (configure(primary) != RKNN_SUCC) {
        cleanup();
        return false;
    }

    for (size_t index = 1; index < context_count; ++index) {
        rknn_context duplicate_context = 0;
        if (duplicate(contexts.front(), duplicate_context) != RKNN_SUCC || duplicate_context == 0) {
            if (duplicate_context != 0) {
                destroy(duplicate_context);
            }
            cleanup();
            return false;
        }
        contexts.push_back(duplicate_context);
        if (configure(duplicate_context) != RKNN_SUCC) {
            cleanup();
            return false;
        }
    }
    return true;
}

RknnExecutionPool::ContextLease::~ContextLease() {
    owner_.release_context(context_);
}

void RknnExecutionPool::worker() {
    while (true) {
        Work         work;
        rknn_context context = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this]() {
                return !running_ || (!jobs_.empty() && !available_contexts_.empty());
            });
            if (!running_ && jobs_.empty()) {
                return;
            }
            if (jobs_.empty() || available_contexts_.empty()) {
                continue;
            }
            work = std::move(jobs_.front());
            jobs_.pop_front();
            context = available_contexts_.front();
            available_contexts_.pop_front();
        }
        ContextLease lease(*this, context);
        try {
            work.task(lease.get());
        } catch (...) {
            cancel_safely(work.cancel);
        }
    }
}

void RknnExecutionPool::release_context(rknn_context context) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        available_contexts_.push_back(context);
    }
    condition_.notify_one();
}

void RknnExecutionPool::cancel_safely(const Cancel& cancel) {
    try {
        cancel();
    } catch (...) {
    }
}

}  // namespace infer
