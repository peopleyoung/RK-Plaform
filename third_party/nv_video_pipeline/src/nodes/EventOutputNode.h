#pragma once

extern "C" {
#include <libavformat/avformat.h>
}

#include <cstdint>
#include <climits>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <string>

#include "PacketMeta.h"
#include "ProcessNode.h"

namespace Node {

class EventOutputNode : public GraphCore::Node {
public:
    explicit EventOutputNode(const std::string& name);
    ~EventOutputNode();
    bool Init(YAML::Node config) override;

private:
    struct BufferedPacket {
        object_meta::PacketMeta::ptr meta;
        object_meta::EncodedPacket::ptr packet;
    };

    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
    void                prune_buffer();
    void                cleanup_retention();
    bool                save_snapshot(const Data::BaseData::ptr& data, const std::string& event_id,
                                      std::filesystem::path& result);
    bool                start_recording(const std::string& event_id);
    bool                write_packet(const BufferedPacket& packet);
    void                stop_recording();
    static std::string  safe_filename(const std::string& value);

    std::string           task_id_;
    std::filesystem::path output_root_;
    bool                  snapshot_enabled_{true};
    bool                  record_enabled_{false};
    int                   pre_seconds_{3};
    int                   post_seconds_{5};
    int                   retention_days_{30};
    std::deque<BufferedPacket> buffer_;
    std::ofstream         event_log_;

    AVFormatContext*      record_format_{nullptr};
    AVStream*             record_stream_{nullptr};
    bool                  record_header_written_{false};
    std::filesystem::path record_path_;
    int64_t               recording_until_us_{0};
    int64_t               record_base_pts_us_{0};
    int64_t               last_written_pts_us_{INT64_MIN};
    std::chrono::steady_clock::time_point last_retention_cleanup_{};
};

}  // namespace Node
