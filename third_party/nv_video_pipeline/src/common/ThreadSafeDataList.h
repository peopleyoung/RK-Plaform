#pragma once

#include <condition_variable>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>

#include "BaseData.h"
#include "CLog.h"

namespace GraphCore {

// 缓冲队列满时的策略
enum BufferOverStrategy {
    DROP_EARLY,  // 丢弃最早的帧
    DROP_LATE,   // 丢弃最新的帧
    CLEAR,       // 清空缓冲队列
    BLOCK        // 堵塞，直到队列有空间
};

// 线程安全队列
class ThreadSafeDataList {
public:
    using ptr = std::shared_ptr<ThreadSafeDataList>;

    ThreadSafeDataList(std::string name) : m_name(name) {
    }

public:
    bool Push(const Data::BaseData::ptr &data) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_list.size() > max_number) {
            switch (m_buffer_strategy) {
                case BufferOverStrategy::DROP_EARLY: {
                    // 缓存队列满了，丢弃最早的帧，保证实时性，但不丢弃其他信息数据
                    m_list.pop_front();
                    m_list.push_back(data);
                    m_work_cond->notify_one();
                    // LOG_WARN("{} buffer full, drop the earliest frame", m_name);
                    return false;  // 返回的false标识缓冲队列满
                }
                case BufferOverStrategy::DROP_LATE: {
                    m_list.pop_back();
                    m_list.push_back(data);
                    m_work_cond->notify_one();
                    return false;  // 丢弃的是最新的帧
                }
                case BufferOverStrategy::CLEAR: {
                    m_list.clear();
                    m_list.push_back(data);
                    m_work_cond->notify_one();
                    return false;
                }
                case BufferOverStrategy::BLOCK: {
                    m_self_cond.wait(lock);
                    break;
                }
                default: {
                    LOG_ERROR("unknown buffer over strategy");
                }
            }
        } else {
            { m_list.push_back(data); }
            m_work_cond->notify_one();
        }
        return true;
    }

    bool Pop(Data::BaseData::ptr &data) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_list.empty()) {
            return false;
        }

        data = m_list.front();
        m_list.pop_front();

        if (m_buffer_strategy == BufferOverStrategy::BLOCK) {
            m_self_cond.notify_one();
        }
        return true;
    }

    // PopList函数，从队列中取出最多num个数据
    bool PopList(std::vector<Data::BaseData::ptr> &data_list, int max_num) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_list.empty()) {
            return false;
        }

        for (int i = 0; i < max_num; i++) {
            if (m_list.empty()) {
                break;
            }
            data_list.push_back(m_list.front());
            m_list.pop_front();
        }

        if (m_buffer_strategy == BufferOverStrategy::BLOCK) {
            m_self_cond.notify_one();
        }
        return true;
    }

    bool push_front(const Data::BaseData::ptr &data) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_list.size() > max_number) {
            switch (m_buffer_strategy) {
                case BufferOverStrategy::DROP_EARLY: {
                    // 缓存队列满了，丢弃最早的帧，保证实时性，但不丢弃其他信息数据
                    m_list.pop_front();
                    m_list.push_front(data);
                    m_work_cond->notify_one();
                    // LOG_WARN("{} buffer full, drop the earliest frame", m_name);
                    return false;  // 返回的false标识缓冲队列满
                }
                case BufferOverStrategy::DROP_LATE: {
                    m_list.pop_back();
                    m_list.push_front(data);
                    m_work_cond->notify_one();
                    return false;  // 丢弃的是最新的帧
                }
                case BufferOverStrategy::CLEAR: {
                    m_list.clear();
                    m_list.push_front(data);
                    m_work_cond->notify_one();
                    return false;
                }
                case BufferOverStrategy::BLOCK: {
                    m_self_cond.wait(lock);
                    break;
                }
                default: {
                    LOG_ERROR("unknown buffer over strategy");
                    exit(EXIT_FAILURE);
                }
            }
        } else {
            { m_list.push_front(data); }
            m_work_cond->notify_one();
        }
        return true;
    }

    void set_max_size(const int size) {
        std::unique_lock<std::mutex> lock(m_mutex);
        max_number = size;
    }

    int size() {
        std::unique_lock<std::mutex> lock(m_mutex);
        return (int)m_list.size();
    }

    void clear() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_list.clear();
    }

    void setCond(std::shared_ptr<std::condition_variable> &cond) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_work_cond = cond;
    }

    void set_buffer_strategy(BufferOverStrategy strategy) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_buffer_strategy = strategy;
    }

private:
    std::mutex                               m_mutex;
    std::shared_ptr<std::condition_variable> m_work_cond;                   // 用于唤醒工作线程的条件变量
    std::condition_variable                  m_self_cond;                   // 用于唤醒自身的条件变量
    std::list<Data::BaseData::ptr>           m_list;                        // 缓冲队列
    size_t                                   max_number = 25;               // 默认最大缓冲帧数
    BufferOverStrategy m_buffer_strategy = BufferOverStrategy::DROP_EARLY;  // 缓冲队列满时的策略
    std::string        m_name;
};

}  // namespace GraphCore
