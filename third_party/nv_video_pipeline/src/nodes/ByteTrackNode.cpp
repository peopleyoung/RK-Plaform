#include "ByteTrackNode.h"

#include "BaseData.h"
#include "CLog.h"
#include "FrameBoxTarget.h"
#include "ProcessNode.h"
#include "Register.h"

namespace Node {

bool ByteTrackNode::Init(YAML::Node config) {
    if (!Node::Init(config)) {
        return false;
    }
    m_type = GraphCore::MID_NODE;
    if (config["track_buffer"]) {
        m_track_buffer = config["track_buffer"].as<int>();
    }
    if (config["frame_rate"]) {
        m_fps = config["frame_rate"].as<int>();
    }
    if (m_track_buffer < 1 || m_track_buffer > 10000 || m_fps < 1 || m_fps > 240) {
        LOG_ERROR("ByteTrackNode {} has invalid frame rate or track buffer", getName());
        return false;
    }
    return true;
}

Data::BaseData::ptr ByteTrackNode::handle_data(Data::BaseData::ptr data) {
    const auto packet_meta = data->get_packet_meta();
    const int  fps         = packet_meta && packet_meta->fps > 0 ? packet_meta->fps : m_fps;
    if (fps != m_fps) {
        m_fps = fps;
        for (auto& track : trackers) {
            track.second->reset_parameters(m_fps, m_track_buffer);
        }
    }

    for (auto& it : data->target_map) {
        auto                           key     = it.first;
        auto                           targets = it.second;
        std::vector<ByteTrack::Object> objects;
        for (auto& target : targets->targets) {
            auto box = std::dynamic_pointer_cast<object_meta::FrameBoxTarget>(target);
            if (!box) {
                continue;
            }
            ByteTrack::Object obj;
            obj.rect.x       = box->left;
            obj.rect.y       = box->top;
            obj.rect.width   = box->right - box->left;
            obj.rect.height  = box->bottom - box->top;
            obj.frame_target = target;
            obj.score        = box->confidence;
            objects.push_back(obj);
        }

        if (trackers.find(key) == trackers.end()) {
            trackers[key] = std::make_shared<ByteTrack::BYTETracker>(m_fps, m_track_buffer);
        }
        auto output_stracks = trackers[key]->update(objects);

        targets->targets.clear();
        for (auto& strack : output_stracks) {
            auto box = std::dynamic_pointer_cast<object_meta::FrameBoxTarget>(strack.frame_target);
            if (!box) {
                continue;
            }
            box->track_id = strack.track_id;
            box->left     = strack.tlwh[0];
            box->top      = strack.tlwh[1];
            box->right    = strack.tlwh[0] + strack.tlwh[2];
            box->bottom   = strack.tlwh[1] + strack.tlwh[3];
            targets->targets.push_back(box);
        }
    }

    return data;
}

}  // namespace Node

namespace {
Register<GraphCore::Node, Node::ByteTrackNode, std::string> _("ByteTrackNode");
}
