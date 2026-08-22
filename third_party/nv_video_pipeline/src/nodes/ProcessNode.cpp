#include "ProcessNode.h"

#include <cmath>
#include <memory>
#include <utility>

#include "BaseData.h"
#include "CLog.h"

namespace GraphCore {

const std::string Node::getName() {
    return m_name;
}

Node::~Node() {
}

void Node::start() {
    if (!m_run) {
        if (m_type == SRC_NODE && m_input_buffers.empty()) {
            auto input_queue = std::make_shared<GraphCore::ThreadSafeDataList>(getName());
            input_queue->set_max_size(200);
            input_queue->set_buffer_strategy(GraphCore::BufferOverStrategy::DROP_EARLY);
            add_input("input", input_queue);
        }

        m_run    = true;
        m_worker = std::thread(&Node::worker, this);

        // todo 这里需要等待线程启动完成才能同步执行
        if (after_start_cb) {
            after_start_cb();
        }
        LOG_DEBUG("{} 节点线程启动完成", getName());
    } else {
        LOG_ERROR("该线程重复启动");
    }
}

void Node::stop() {
    m_run = false;
    std::for_each(m_input_buffers.begin(), m_input_buffers.end(), [&](const auto &item) {
        item.second->clear();
    });
    std::for_each(m_output_buffers.begin(), m_output_buffers.end(), [&](const auto &item) {
        item.second->clear();
    });
    m_base_cond->notify_all();
    if (m_worker.joinable())
        m_worker.join();
    if (exit_cb) {
        exit_cb();
    }
    LOG_DEBUG("{} 节点线程退出完成", getName());
}

void Node::add_input(const std::string &tag, QUEUE queue) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    queue->setCond(m_base_cond);
    m_input_buffers.insert(make_pair(tag, queue));
}

void Node::add_output(const std::string &tag, QUEUE queue) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    m_output_buffers.insert(make_pair(tag, queue));
}

void Node::del_input(const std::string &tag) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_input_buffers.find(tag) != m_input_buffers.end()) {
        m_input_buffers.erase(tag);
    }
}

void Node::del_output(const std::string &tag) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_output_buffers.find(tag) != m_output_buffers.end()) {
        m_output_buffers.erase(tag);
    }
}

void Node::worker() {
    if (before_start_cb) {
        if (before_start_cb() != 0) {
            LOG_ERROR("{}节点初始化失败", getName());
            exit(EXIT_FAILURE);
        }
    }
    std::vector<Data::BaseData::ptr> datas;
    while (m_run) {
        get_input_datas(datas);

        if (!datas.empty()) {
            m_ticker.resetTime();
            if (batch_data_handler_hooker) {
                auto res = batch_data_handler_hooker(datas);
                send_output_datas(res);
                continue;
            }
            for (auto &data : datas) {
                if (m_before_data_handle_callback) {
                    data = m_before_data_handle_callback(data);
                }
                data = handle_data(data);
                if (m_after_data_handle_callback) {
                    data = m_after_data_handle_callback(data);
                }
                send_output_data(data);
            }

            std::unique_lock<std::mutex> lk(m_perf_mutex);
            m_perf_time += m_ticker.elapsedTime();
            m_perf_count += datas.size();

        } else {
            std::unique_lock<std::mutex> lk(m_base_mutex);
            // A timed wait also closes the check/wait lost-wakeup window in the legacy queue API.
            m_base_cond->wait_for(lk, std::chrono::milliseconds(100));
        }
    }
}

std::tuple<double, float> Node::get_perf() {
    std::unique_lock<std::mutex> lk(m_perf_mutex);
    auto                         perf = std::make_tuple(m_perf_time, m_perf_count);
    m_perf_time                       = 0;
    m_perf_count                      = 0;
    return perf;
}

void Node::get_input_datas(std::vector<Data::BaseData::ptr> &datas) {
    datas.clear();

    for (auto &item : m_input_buffers) {
        item.second->PopList(datas, m_get_data_max_num);
    }
}

void Node::send_output_data(const Data::BaseData::ptr &data) {
    if (!data) {
        return;
    }
    for (auto &item : m_output_buffers) {
        if (!item.second->Push(data)) {
            m_buffer_over_count++;
            if (buffer_over_cb) {
                buffer_over_cb();
            }
        }
    }
}

void Node::send_output_datas(const std::vector<Data::BaseData::ptr> &datas) {
    for (auto &data : datas) {
        send_output_data(data);
    }
}

Data::BaseData::ptr Node::handle_data(Data::BaseData::ptr data) {
    return data;
}

void Node::add_data(const Data::BaseData::ptr &data) {
    m_input_buffers.begin()->second->Push(data);
}

void Node::add_datas(const std::vector<Data::BaseData::ptr> &datas) {
    for (auto &data : datas) {
        add_data(data);
    }
}

NODE_TYPE Node::getType() {
    return m_type;
}

void Node::set_get_data_max_num(int num) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    m_get_data_max_num = num;
}

void Node::set_before_data_handle_callback(DataHandleCallBack callback) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    m_before_data_handle_callback = std::move(callback);
}

void Node::set_after_data_handle_callback(DataHandleCallBack callback) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    m_after_data_handle_callback = std::move(callback);
}

void Node::set_extra_input_callback(Node::ExtraInputCallBack callback) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    m_extra_input_callback = std::move(callback);
}

void Node::set_batch_data_handler_hooker(BatchDataHookerFunc batch_data_hooker) {
    batch_data_handler_hooker = std::move(batch_data_hooker);
}

void Node::add_extra_data(const Data::BaseData::ptr &data) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_extra_input_callback) {
        m_extra_input_callback(data);
    }
}

bool Node::Init(YAML::Node) {
    set_buffer_over_cb([this]() {
        if (m_buffer_over_count % 50 == 0) {
            LOG_WARN("{} 节点缓冲区溢出", getName());
        }
        m_buffer_over_count = 0;
        return 0;
    });
    return true;
}

}  // namespace GraphCore
