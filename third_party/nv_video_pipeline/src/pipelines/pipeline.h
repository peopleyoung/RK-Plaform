#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Poller/Timer.h"
#include "ProcessNode.h"

namespace Global {
inline std::atomic<float> PIPE_PERF_INTERVAL = 5.0f;
}  // namespace Global

class Pipeline {
public:
    using ptr                = std::shared_ptr<Pipeline>;
    using ExtraInputCallBack = std::function<void(Data::BaseData::ptr)>;
    using DataHandleCallBack = std::function<Data::BaseData::ptr(Data::BaseData::ptr)>;

public:
    Pipeline() = default;

    explicit Pipeline(const std::string& task_name) : m_task_name(task_name) {
    }

    ~Pipeline();

    const std::string get_name() const {
        return m_task_name;
    }

public:
    void add_data(std::string node_name, Data::BaseData::ptr data);
    void start();
    void stop();
    bool init(const std::string& config_file, int index);

private:
    bool config_check();
    bool perf_timer();

private:
    int                               m_index;
    std::string                       m_input_node_name;
    YAML::Node                        m_config_root;
    std::string                       m_input_url;
    std::vector<GraphCore::Node::ptr> m_perf_nodes;
    bool                              m_running = false;
    toolkit::Timer::Ptr               m_timer;

protected:
    std::mutex                                          m_mutex;
    const std::string                                   m_task_name;
    std::map<std::string, GraphCore::Node::ptr>         m_nodes;
    std::unordered_map<std::string, DataHandleCallBack> m_before_callbacks;
    std::unordered_map<std::string, DataHandleCallBack> m_after_callbacks;
};
