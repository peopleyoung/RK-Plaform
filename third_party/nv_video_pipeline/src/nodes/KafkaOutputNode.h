#pragma once

#include <librdkafka/rdkafka.h>

#include <cstdint>
#include <string>

#include "ProcessNode.h"

namespace Node {

class KafkaOutputNode : public GraphCore::Node {
public:
    explicit KafkaOutputNode(const std::string& name);
    ~KafkaOutputNode();

    bool Init(YAML::Node config) override;

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
    static void         delivery_report(rd_kafka_t*, const rd_kafka_message_t* message, void* opaque);

    rd_kafka_t* producer_{nullptr};
    std::string topic_{"sei_msg"};
    std::string key_;
    std::string task_id_;
    std::string instance_name_;
    uint64_t    revision_{0};
    uint64_t    delivered_{0};
    uint64_t    failed_{0};
};

}  // namespace Node
