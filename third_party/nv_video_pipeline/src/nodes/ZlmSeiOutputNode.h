#pragma once

extern "C" {
#include <libavformat/avformat.h>
}

#include <chrono>
#include <cstdint>
#include <string>

#include "PacketMeta.h"
#include "ProcessNode.h"
#include "TimestampNormalizer.h"

namespace Node {

class ZlmSeiOutputNode : public GraphCore::Node {
public:
    explicit ZlmSeiOutputNode(const std::string& name);
    ~ZlmSeiOutputNode();

    bool Init(YAML::Node config) override;

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
    bool                open(const object_meta::PacketMeta::ptr& packet);
    void                close();
    void                schedule_retry(std::chrono::steady_clock::time_point now);
    bool                metadata_within_limits(const Data::BaseData::ptr& data) const;
    void                warn_sei_skipped();

    std::string output_uri_;
    int         reconnect_ms_{1000};
    std::string task_id_;
    std::string instance_name_;
    uint64_t    revision_{0};
    AVFormatContext* format_{nullptr};
    AVStream*        stream_{nullptr};
    bool             header_written_{false};
    object_meta::VideoCodec codec_{object_meta::VideoCodec::H264};
    std::chrono::steady_clock::time_point retry_after_{};
    std::chrono::steady_clock::time_point last_sei_warning_{};
    uint32_t         consecutive_failures_{0};
    uint64_t         publish_failure_count_{0};
    uint64_t         sei_skipped_count_{0};
    media::TimestampNormalizer timestamp_normalizer_;
};

}  // namespace Node
