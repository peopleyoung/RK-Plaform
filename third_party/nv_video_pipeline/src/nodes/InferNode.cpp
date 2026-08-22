#include "InferNode.h"

#include <Thread/ThreadPool.h>
#include <Util/util.h>

#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <utility>

#include "BaseData.h"
#include "CLog.h"
#include "FrameTarget.h"
#include "InstancesManager.h"
#include "ProcessNode.h"
#include "Register.h"
#include "StatusCode.h"
#include "Thread/ThreadPool.h"

namespace Node {

InferNode::InferNode(const std::string& name) : Node(std::move(name)) {
    m_type = GraphCore::MID_NODE;
    pool   = std::make_shared<toolkit::ThreadPool>(1, toolkit::ThreadPool::PRIORITY_HIGHEST, true);
    set_after_start_cb(std::bind(&InferNode::after_start_cb, this));
    set_exit_cb(std::bind(&InferNode::exit_cb, this));
}

void InferNode::set_instance(infer::Instance::ptr instance) {
    m_instance = std::move(instance);
}

bool InferNode::Init(YAML::Node config) {
    CHECK(config["instance"], toolkit::str_format("%s节点未指定推理实例！", getName().c_str()));

    m_instance_name = config["instance"].as<std::string>();
    CHECK(InstancesManager::get()->has_key(m_instance_name),
          toolkit::str_format("%s节点未找到推理实例：%s", getName().c_str(), m_instance_name.c_str()));

    set_instance(InstancesManager::get()->get_instance(m_instance_name));
    if (config["interval"]) {
        auto interval = config["interval"].as<int>();
        if (interval <= 0) {
            LOG_WARN("interval必须大于等于1，目前配置为{}，将使用默认值1", interval);
            return true;
        }
        if (interval > 1 && !m_instance->supports_interval_reuse()) {
            LOG_ERROR("InferNode {} interval={} is unsupported for structured inference instance {}", getName(),
                      interval, m_instance_name);
            return false;
        }
        m_interval = interval;
    }

    return true;
}

Data::BaseData::ptr InferNode::handle_data_interval(Data::BaseData::ptr data) {
    if (m_target_in_queue == 0) {
        Job job;
        job.data = data;
        if (!m_instance->commit(job)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            m_commit_error_count++;
            if (m_commit_error_count % 50 == 0) {
                LOG_WARN("InferNode: {} 推送推理任务失败", getName());
                m_commit_error_count = 0;
            }
            return nullptr;
        }

        pool->async([this, job]() {
            m_target_in_queue += m_interval - 1;
            auto future = job.promise->get_future();
            if (future_wait_for_true(future, 5000) != true) {
                LOG_WARN("InferNode: {} 推理超时", getName());
                for (int i = 0; i < m_interval - 1; i++) {
                    m_target_queue->push(nullptr);
                }
                return;
            }
            auto job_data = job.data;
            if (m_after_data_handle_callback) {
                job_data = m_after_data_handle_callback(job_data);
            }
            for (int i = 0; i < m_interval - 1; i++) {
                auto target_copy = job_data->get_frame_target_list(m_instance_name)->deep_copy();
                m_target_queue->push(target_copy);
            }
            send_output_data(job_data);
        });
    } else {
        pool->async([this, data]() {
            object_meta::FrameTargetList::ptr target;
            m_target_queue->wait_and_pop(target);
            if (target != nullptr) {
                data->set_frame_target_list(m_instance_name, target);
                send_output_data(data);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        });
        m_target_in_queue--;
    }
    return nullptr;
}

Data::BaseData::ptr InferNode::handle_data_normal(Data::BaseData::ptr data) {
    Job job;
    job.data = data;
    if (!m_instance->commit(job)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        m_commit_error_count++;
        if (m_commit_error_count % 50 == 0) {
            LOG_WARN("InferNode: {} 推送推理任务失败", getName());
            m_commit_error_count = 0;
        }
        return nullptr;
    }
    pool->async([this, job]() {
        auto future = job.promise->get_future();
        if (future_wait_for_true(future, 5000) != true) {
            LOG_WARN("InferNode: {} 推理超时", getName());
            return;
        }
        auto job_data = job.data;
        if (m_after_data_handle_callback) {
            job_data = m_after_data_handle_callback(job_data);
        }
        send_output_data(job_data);
    });
    return nullptr;
}

Data::BaseData::ptr InferNode::handle_data(Data::BaseData::ptr data) {
    if (m_instance == nullptr) {
        LOG_ERROR("InferNode: {} is nullptr", getName());
        exit(EXIT_FAILURE);
        return nullptr;
    }
    if (m_interval > 1) {
        return handle_data_interval(std::move(data));
    }
    return handle_data_normal(std::move(data));
}

int InferNode::exit_cb() {
    pool.reset();
    m_target_queue.reset();
    return 0;
}

int InferNode::after_start_cb() {
    m_target_queue = utils::ThreadsafeQueue<object_meta::FrameTargetList::ptr>::createShared(0);
    return 0;
}

}  // namespace Node

namespace {
Register<GraphCore::Node, Node::InferNode, std::string> _("InferNode");
}
