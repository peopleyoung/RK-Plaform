#pragma once

#include <curl/curl.h>

#include <fstream>
#include <mutex>
#include <string>

#include "ProcessNode.h"

namespace Node {

class JsonOutputNode : public GraphCore::Node {
public:
    explicit JsonOutputNode(const std::string& name);
    ~JsonOutputNode();
    bool Init(YAML::Node config) override;

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
    bool                emit(const std::string& line);

    std::string   instance_name_;
    std::string   task_id_;
    uint64_t      revision_{0};
    std::ofstream output_;
    std::string   http_url_;
    std::string   authorization_;
    long          connect_timeout_ms_{1000};
    long          request_timeout_ms_{3000};
    CURL*         curl_{nullptr};
    std::mutex    output_mutex_;
};

}  // namespace Node
