#pragma once

#include <memory>
#include <unordered_map>

#include "ProcessNode.h"
#include "track/bytetrack/ByteTracker.hpp"

namespace Node {
class ByteTrackNode : public GraphCore::Node {
public:
    using ptr = std::shared_ptr<ByteTrackNode>;

    explicit ByteTrackNode(const std::string& name) : GraphCore::Node(name) {
        m_type = GraphCore::MID_NODE;
    }

    bool Init(YAML::Node config) override;

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
    std::unordered_map<std::string, std::shared_ptr<ByteTrack::BYTETracker>> trackers;
    int                                                                      m_fps          = 30;
    int                                                                      m_track_buffer = 30;
};
}  // namespace Node
