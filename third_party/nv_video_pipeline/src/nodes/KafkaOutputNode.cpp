#include "KafkaOutputNode.h"

#include <algorithm>

#include "AnhuanMessage.h"
#include "CLog.h"
#include "Register.h"
#include "StatusCode.h"

namespace Node {

KafkaOutputNode::KafkaOutputNode(const std::string& name) : GraphCore::Node(name) {
    m_type = GraphCore::DES_NODE;
}

KafkaOutputNode::~KafkaOutputNode() {
    if (producer_) {
        rd_kafka_flush(producer_, 3000);
        rd_kafka_destroy(producer_);
    }
}

void KafkaOutputNode::delivery_report(rd_kafka_t*, const rd_kafka_message_t* message, void* opaque) {
    auto* self = static_cast<KafkaOutputNode*>(opaque);
    if (message->err) {
        ++self->failed_;
        if (self->failed_ == 1 || self->failed_ % 100 == 0) {
            LOG_WARN("KafkaOutputNode {} delivery failed {} time(s): {}", self->getName(), self->failed_,
                     rd_kafka_err2str(message->err));
        }
    } else {
        ++self->delivered_;
    }
}

bool KafkaOutputNode::Init(YAML::Node config) {
    CHECK(GraphCore::Node::Init(config), "KafkaOutputNode base initialization failed");
    CHECK(config["brokers"], "KafkaOutputNode requires brokers");
    const std::string brokers = config["brokers"].as<std::string>();
    CHECK(!brokers.empty(), "KafkaOutputNode brokers cannot be empty");
    if (config["topic"]) {
        topic_ = config["topic"].as<std::string>();
    }
    CHECK(!topic_.empty(), "KafkaOutputNode topic cannot be empty");
    key_ = config["key"] ? config["key"].as<std::string>()
                         : default_anhuan_key(config["input"].as<std::string>());
    task_id_       = config["task_id"] ? config["task_id"].as<std::string>() : "";
    instance_name_ = config["instance"] ? config["instance"].as<std::string>() : "";
    revision_      = config["revision"] ? config["revision"].as<uint64_t>() : 0;

    auto* conf  = rd_kafka_conf_new();
    const auto set_conf = [&](const char* name, const std::string& value) {
        char message[512] = {};
        if (rd_kafka_conf_set(conf, name, value.c_str(), message, sizeof(message)) != RD_KAFKA_CONF_OK) {
            LOG_ERROR("KafkaOutputNode {} invalid {}: {}", getName(), name, message);
            return false;
        }
        return true;
    };
    const int queue_messages = config["queue_messages"] ? config["queue_messages"].as<int>() : 10000;
    const int message_timeout = config["message_timeout_ms"] ? config["message_timeout_ms"].as<int>() : 3000;
    CHECK(queue_messages >= 1 && queue_messages <= 1000000, "Kafka queue_messages is out of range");
    CHECK(message_timeout >= 100 && message_timeout <= 60000, "Kafka message_timeout_ms is out of range");
    CHECK(set_conf("bootstrap.servers", brokers), "Kafka brokers configuration failed");
    CHECK(set_conf("queue.buffering.max.messages", std::to_string(queue_messages)), "Kafka queue limit failed");
    CHECK(set_conf("message.timeout.ms", std::to_string(message_timeout)), "Kafka timeout failed");
    CHECK(set_conf("enable.idempotence", "true"), "Kafka idempotence failed");
    rd_kafka_conf_set_dr_msg_cb(conf, &KafkaOutputNode::delivery_report);
    rd_kafka_conf_set_opaque(conf, this);
    char error_buffer[512] = {};
    producer_ = rd_kafka_new(RD_KAFKA_PRODUCER, conf, error_buffer, sizeof(error_buffer));
    if (!producer_) {
        LOG_ERROR("KafkaOutputNode {} could not create producer: {}", getName(), error_buffer);
        return false;
    }
    LOG_INFO("KafkaOutputNode {} configured topic {} key {}", getName(), topic_, key_);
    return true;
}

Data::BaseData::ptr KafkaOutputNode::handle_data(Data::BaseData::ptr data) {
    const std::string message = build_anhuan_message(data, task_id_, revision_, instance_name_);
    const auto result = rd_kafka_producev(
        producer_, RD_KAFKA_V_TOPIC(topic_.c_str()), RD_KAFKA_V_KEY(key_.data(), key_.size()),
        RD_KAFKA_V_VALUE(const_cast<char*>(message.data()), message.size()), RD_KAFKA_V_MSGFLAGS(RD_KAFKA_MSG_F_COPY),
        RD_KAFKA_V_END);
    if (result != RD_KAFKA_RESP_ERR_NO_ERROR) {
        ++failed_;
        if (failed_ == 1 || failed_ % 100 == 0) {
            LOG_WARN("KafkaOutputNode {} queue rejected {} message(s): {}", getName(), failed_,
                     rd_kafka_err2str(result));
        }
    }
    rd_kafka_poll(producer_, 0);
    return data;
}

}  // namespace Node

namespace {
Register<GraphCore::Node, Node::KafkaOutputNode, std::string> register_kafka_output("KafkaOutputNode");
}
