#include "RknnExecutionPool.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

void test_workers_lease_contexts_exclusively() {
    std::mutex                       state_mutex;
    std::map<rknn_context, int>      active;
    std::set<rknn_context>           seen;
    std::atomic_int                  processed{0};
    std::vector<std::promise<void>>  completions(18);
    std::vector<std::future<void>>   futures;
    std::vector<rknn_context>        destroyed;

    infer::RknnExecutionPool pool;
    assert(pool.init({11, 22, 33}, 2, 36, [&](rknn_context context) {
        std::lock_guard<std::mutex> lock(state_mutex);
        destroyed.push_back(context);
    }));
    assert(pool.context_count() == 3);
    assert(pool.worker_count() == 2);
    pool.start();

    for (size_t index = 0; index < completions.size(); ++index) {
        futures.push_back(completions[index].get_future());
        assert(pool.commit(
            [&, index](rknn_context context) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    assert(active[context] == 0);
                    active[context] += 1;
                    seen.insert(context);
                }
                std::this_thread::sleep_for(2ms);
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    active[context] -= 1;
                }
                processed.fetch_add(1);
                completions[index].set_value();
            },
            [&completions, index]() {
                completions[index].set_exception(
                    std::make_exception_ptr(std::runtime_error("cancelled")));
            }));
    }
    for (auto& future : futures) {
        assert(future.wait_for(2s) == std::future_status::ready);
        future.get();
    }
    pool.stop();

    assert(processed.load() == static_cast<int>(completions.size()));
    assert(seen == std::set<rknn_context>({11, 22, 33}));
    assert(std::set<rknn_context>(destroyed.begin(), destroyed.end()) == seen);
    assert(destroyed.size() == 3);
}

void test_failure_returns_context_and_worker_continues() {
    std::promise<void> cancelled;
    std::promise<void> completed;
    std::atomic_int    destroy_count{0};

    infer::RknnExecutionPool pool;
    assert(pool.init({44}, 1, 4, [&](rknn_context) {
        destroy_count.fetch_add(1);
    }));
    pool.start();
    assert(pool.commit(
        [](rknn_context) {
            throw std::runtime_error("expected test failure");
        },
        [&]() {
            cancelled.set_value();
        }));
    assert(pool.commit(
        [&](rknn_context context) {
            assert(context == 44);
            completed.set_value();
        },
        []() {}));

    assert(cancelled.get_future().wait_for(2s) == std::future_status::ready);
    assert(completed.get_future().wait_for(2s) == std::future_status::ready);
    pool.stop();
    assert(destroy_count.load() == 1);
}

void test_stop_cancels_queued_work_before_destroy() {
    std::mutex              mutex;
    std::condition_variable condition;
    bool                    active_started = false;
    bool                    release_active = false;
    std::promise<void>      queued_cancelled;
    std::atomic_bool        destroyed{false};

    infer::RknnExecutionPool pool;
    assert(pool.init({55}, 1, 4, [&](rknn_context) {
        destroyed.store(true);
    }));
    pool.start();
    assert(pool.commit(
        [&](rknn_context) {
            std::unique_lock<std::mutex> lock(mutex);
            active_started = true;
            condition.notify_all();
            condition.wait(lock, [&]() {
                return release_active;
            });
        },
        []() {}));
    assert(pool.commit([](rknn_context) {}, [&]() {
        queued_cancelled.set_value();
    }));
    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&]() {
            return active_started;
        });
    }
    std::thread stopper([&]() {
        pool.stop();
    });
    assert(queued_cancelled.get_future().wait_for(2s) == std::future_status::ready);
    assert(!destroyed.load());
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_active = true;
    }
    condition.notify_all();
    stopper.join();
    assert(destroyed.load());
}

void test_partial_context_creation_cleans_up() {
    std::vector<rknn_context> destroyed;
    std::vector<rknn_context> contexts;
    int                       duplicate_calls = 0;
    const bool created = infer::RknnExecutionPool::create_contexts(
        3,
        [](rknn_context& context) {
            context = 101;
            return RKNN_SUCC;
        },
        [&](rknn_context&, rknn_context& context) {
            duplicate_calls += 1;
            if (duplicate_calls == 2) {
                return -1;
            }
            context = 202;
            return RKNN_SUCC;
        },
        [](rknn_context) {
            return RKNN_SUCC;
        },
        [&](rknn_context context) {
            destroyed.push_back(context);
        },
        contexts);

    assert(!created);
    assert(contexts.empty());
    assert(destroyed == std::vector<rknn_context>({101, 202}));
}

}  // namespace

int main() {
    test_workers_lease_contexts_exclusively();
    test_failure_returns_context_and_worker_continues();
    test_stop_cancels_queued_work_before_destroy();
    test_partial_context_creation_cleans_up();
    return 0;
}
