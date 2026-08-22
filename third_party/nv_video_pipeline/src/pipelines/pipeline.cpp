#include "pipeline.h"

#include <Util/util.h>
#include <json/json.h>

#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "CLog.h"
#include "ProcessNode.h"

void Pipeline::start() {
    for (auto& node : m_nodes) {
        if (node.first != m_input_node_name) {
            node.second->start();
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    m_nodes[m_input_node_name]->start();
    m_running = true;
    m_timer =
        std::make_shared<toolkit::Timer>(Global::PIPE_PERF_INTERVAL, std::bind(&Pipeline::perf_timer, this), nullptr);
}

void Pipeline::stop() {
    m_running = false;
    m_timer.reset();
    for (auto& node : m_nodes) {
        node.second->stop();
    }
}

Pipeline::~Pipeline() {
    // stop();
}

bool Pipeline::perf_timer() {
    double perf_time_cost = 0;
    float  perf_count     = 0;

    Json::Value perf_data;
    for (auto& node : m_nodes) {
        std::tie(perf_time_cost, perf_count) = node.second->get_perf();
        if (std::find(m_perf_nodes.begin(), m_perf_nodes.end(), node.second) != m_perf_nodes.end()) {
            if (perf_count == 0) {
                perf_data[node.first] = "0.00ms 0.00fps";
            } else {
                perf_data[node.first] = toolkit::str_format("%.2fms %.2ffps", perf_time_cost / perf_count,
                                                            perf_count / Global::PIPE_PERF_INTERVAL);
            }
        }
    }
    if (m_perf_nodes.size() > 0) {
        Json::FastWriter writer;
        writer.omitEndingLineFeed();
        std::string json_str = writer.write(perf_data);
        LOG_INFO("{}: {}", m_input_url, json_str);
    }
    return m_running;
}

void Pipeline::add_data(std::string node_name, Data::BaseData::ptr data) {
    if (m_nodes.find(node_name) == m_nodes.end()) {
        LOG_ERROR("cannot find node");
        exit(EXIT_FAILURE);
    }
    auto node = m_nodes[node_name];
    node->add_data(data);
}

bool Pipeline::config_check() {
    if (!m_config_root["inputs"]) {
        LOG_ERROR("config file should have inputs");
        return false;
    }
    auto inputs = m_config_root["inputs"];
    m_config_root.remove("inputs");

    m_input_url = inputs[m_index].as<std::string>();

    // 删除不需要的节点
    std::vector<std::string> nodes_to_remove;
    for (auto it = m_config_root.begin(); it != m_config_root.end(); ++it) {
        auto node_name = it->first.as<std::string>();
        auto config    = m_config_root[node_name];
        if (config["enable"]) {
            if (config["enable"].as<int>() == 0) {
                nodes_to_remove.push_back(node_name);  // 记录要删除的节点
            }
        }
    }
    for (const auto& node_name : nodes_to_remove) {
        m_config_root.remove(node_name);
    }

    for (auto it = m_config_root.begin(); it != m_config_root.end(); ++it) {
        auto node_name = it->first.as<std::string>();

        auto node_config = m_config_root[node_name];
        if (!node_config["node"]) {
            LOG_ERROR("{}没有指定node类型", node_name);
            return false;
        }
        if (node_config["link_to"]) {
            if (!node_config["link_to"].IsSequence()) {
                LOG_ERROR("{}上游节点必须为列表", node_name);
                return false;
            }
            if (node_config["link_to"].size() > 1) {
                LOG_ERROR("节点{}配置了多个上游节点，RK3588流水线不支持多输入同步", node_name);
                return false;
            }
        }
    }

    return true;
}

bool Pipeline::init(const std::string& config_file, int index) {
    m_index       = index;
    m_config_root = YAML::LoadFile(config_file);
    if (!config_check()) {
        return false;
    }

    // 实例化节点
    std::vector<std::string> node_names;
    for (auto it = m_config_root.begin(); it != m_config_root.end(); ++it) {
        auto node_name   = it->first.as<std::string>();
        auto node_config = m_config_root[node_name];
        auto node_class  = node_config["node"].as<std::string>();

        auto node          = GraphCore::NodesFactory::Instance().Create(node_class, node_name);
        m_nodes[node_name] = node;

        if (node->getType() == GraphCore::SRC_NODE) {
            m_input_node_name = node_name;
        }
    }

    // 初始化节点
    for (auto it = m_nodes.begin(); it != m_nodes.end(); ++it) {
        auto node_name   = it->first;
        auto node_config = m_config_root[node_name];
        auto node        = it->second;
        if (node_config["outputs"]) {
            node_config["output"] = node_config["outputs"][index].as<std::string>();
        }
        node_config["input"] = m_input_url;
        node_config["index"] = index;
        if (!node->Init(node_config)) {
            LOG_ERROR("init {} node failed", node_name);
            m_nodes.clear();
            return false;
        }
    }

    // 配置需要性能监控的节点
    for (auto it = m_nodes.begin(); it != m_nodes.end(); ++it) {
        auto node_name   = it->first;
        auto node_config = m_config_root[node_name];
        if (node_config["perf"]) {
            if (node_config["perf"].as<int>() == 1) {
                m_perf_nodes.push_back(m_nodes[node_name]);
            }
        }
    }

    // 连接节点
    for (auto it = m_nodes.begin(); it != m_nodes.end(); ++it) {
        auto node_name   = it->first;
        auto node_config = m_config_root[node_name];
        auto node        = it->second;
        if (node->getType() == GraphCore::SRC_NODE)
            continue;

        if (!node_config["link_to"]) {
            LOG_ERROR("{} 节点需要上游节点", node_name);
            return false;
        }
        auto link_nodes = node_config["link_to"];
        for (auto link_node_name : link_nodes) {
            auto name = link_node_name.as<std::string>();
            if (m_nodes.find(name) == m_nodes.end()) {
                LOG_ERROR("{}没有找到上游节点{}", node_name, name);
                return false;
            }
            auto link_node = m_nodes[name];
            GraphCore::LinkNode(link_node, node);
        }
    }
    return true;
}
