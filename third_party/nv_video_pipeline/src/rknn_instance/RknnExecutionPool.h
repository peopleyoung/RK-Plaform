#pragma once

#include <rknn_api.h>

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace infer {

class RknnExecutionPool {
public:
    using Task      = std::function<void(rknn_context)>;
    using Cancel    = std::function<void()>;
    using Destroy   = std::function<void(rknn_context)>;
    using Create    = std::function<int(rknn_context&)>;
    using Duplicate = std::function<int(rknn_context&, rknn_context&)>;
    using Configure = std::function<int(rknn_context)>;

    RknnExecutionPool() = default;
    ~RknnExecutionPool();

    RknnExecutionPool(const RknnExecutionPool&)            = delete;
    RknnExecutionPool& operator=(const RknnExecutionPool&) = delete;

    bool init(std::vector<rknn_context> contexts, size_t worker_count, size_t queue_capacity, Destroy destroy);
    bool commit(Task task, Cancel cancel);
    void start();
    void stop();

    size_t context_count() const;
    size_t worker_count() const;

    static bool create_contexts(size_t context_count, Create create, Duplicate duplicate, Configure configure,
                                Destroy destroy, std::vector<rknn_context>& contexts);

private:
    struct Work {
        Task   task;
        Cancel cancel;
    };

    class ContextLease {
    public:
        ContextLease(RknnExecutionPool& owner, rknn_context context) : owner_(owner), context_(context) {
        }
        ~ContextLease();

        rknn_context get() const {
            return context_;
        }

    private:
        RknnExecutionPool& owner_;
        rknn_context       context_;
    };

    void worker();
    void release_context(rknn_context context);
    static void cancel_safely(const Cancel& cancel);

    mutable std::mutex              mutex_;
    std::condition_variable         condition_;
    std::vector<rknn_context>       contexts_;
    std::deque<rknn_context>        available_contexts_;
    std::deque<Work>                jobs_;
    std::vector<std::thread>        workers_;
    Destroy                         destroy_;
    size_t                          configured_worker_count_{0};
    size_t                          queue_capacity_{0};
    bool                            running_{false};
    bool                            accepting_{false};
};

}  // namespace infer
