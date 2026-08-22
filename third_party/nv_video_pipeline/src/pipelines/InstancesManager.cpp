#include "InstancesManager.h"

#include <Util/util.h>
#include <json/json.h>
#include <yaml-cpp/yaml.h>

#include "CLog.h"

bool InstancesManager::has_key(const std::string& key) {
    return m_instances.count(key) > 0;
}

infer::Instance::ptr InstancesManager::get_instance(const std::string& key) {
    return m_instances[key];
}

bool InstancesManager::init(const std::string& config_file) {
    m_config_root = YAML::LoadFile(config_file);
    std::vector<std::string> node_names;
    for (const auto& it : m_config_root) {
        node_names.push_back(it.first.as<std::string>());
    }
    for (auto node_name : node_names) {
        auto sub_config = m_config_root[node_name];
        if (sub_config["enable"]) {
            if (sub_config["enable"].as<int>() == 0)
                continue;
        }
        if (!sub_config["instance_name"]) {
            LOG_ERROR("Instance {} config error, no instance_name", node_name);
            return false;
        }

        auto instance_name = sub_config["instance_name"].as<std::string>();
        auto instance      = infer::InstanceFactory::Instance().Create(instance_name, node_name);
        if (instance->init(sub_config)) {
            LOG_INFO("Instance {} init success", node_name);
        } else {
            LOG_ERROR("Instance {} init failed", node_name);
            return false;
        }
        m_instances[node_name] = instance;

        if (!sub_config["perf"]) {
            m_perf_instances.push_back(instance);
        } else {
            if (sub_config["perf"].as<int>() == 1) {
                m_perf_instances.push_back(instance);
            }
        }
    }
    return true;
}

bool InstancesManager::perf_worker() {
    double perf_time_cost = 0;
    float  perf_count     = 0;

    Json::Value perf_data;
    for (auto& instance : m_instances) {
        std::tie(perf_time_cost, perf_count) = instance.second->get_perf();
        if (std::find(m_perf_instances.begin(), m_perf_instances.end(), instance.second) != m_perf_instances.end()) {
            if (perf_count == 0) {
                perf_data[instance.first] = "0.00ms 0.00fps 0.00ms";
            } else {
                perf_data[instance.first] =
                    // 每帧耗时, 帧率, 每秒耗时
                    toolkit::str_format("%.2fms %.2ffps  %.2fms", perf_time_cost / perf_count,
                                        perf_count / Global::INSTANCE_PERF_INTERVAL,
                                        perf_time_cost / Global::INSTANCE_PERF_INTERVAL);
            }
        }
    }
    if (m_perf_instances.size() > 0) {
        Json::FastWriter writer;
        writer.omitEndingLineFeed();
        std::string json_str = writer.write(perf_data);
        LOG_INFO("{}", json_str);
    }
    return m_running;
}

void InstancesManager::start() {
    for (auto instance : m_instances) {
        instance.second->start();
    }
    m_running           = true;
    m_perf_worker_timer = std::make_shared<toolkit::Timer>(Global::INSTANCE_PERF_INTERVAL,
                                                           std::bind(&InstancesManager::perf_worker, this), nullptr);
}

void InstancesManager::stop() {
    m_running = false;
    m_perf_worker_timer.reset();
    for (auto instance : m_instances) {
        instance.second->stop();
    }
}
