#pragma once

#include <Poller/Timer.h>
#include <yaml-cpp/yaml.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "Instance.h"

namespace Global {
inline std::atomic<float> INSTANCE_PERF_INTERVAL = 5.0f;
}  // namespace Global

using instances_map_type = std::unordered_map<std::string, infer::Instance::ptr>;

class InstancesManager {
public:
    InstancesManager(const InstancesManager&)            = delete;
    InstancesManager& operator=(const InstancesManager&) = delete;

    static InstancesManager* get() {
        static InstancesManager instance;
        return &instance;
    }

    bool                 init(const std::string& config_file);
    void                 start();
    void                 stop();
    bool                 has_key(const std::string& key);
    infer::Instance::ptr get_instance(const std::string& key);

private:
    YAML::Node                        m_config_root;
    instances_map_type                m_instances;
    toolkit::Timer::Ptr               m_perf_worker_timer = nullptr;
    std::vector<infer::Instance::ptr> m_perf_instances;
    bool                              m_running = false;

private:
    InstancesManager()  = default;
    ~InstancesManager() = default;
    bool perf_worker();
};