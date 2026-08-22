#include "AnalyticsNode.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <utility>

#include "BaseData.h"
#include "CLog.h"
#include "FrameBoxTarget.h"
#include "Register.h"
#include "StatusCode.h"

namespace Node {
namespace {

constexpr float side_epsilon = 1e-6F;
constexpr uint32_t stale_track_frames = 300;

std::unordered_set<int> read_class_ids(const YAML::Node& config) {
    std::unordered_set<int> result;
    if (!config || !config.IsSequence()) {
        return result;
    }
    for (const auto& item : config) {
        result.insert(item.as<int>());
    }
    return result;
}

cv::Point2f read_point(const YAML::Node& value) {
    return cv::Point2f(value["x"].as<float>(), value["y"].as<float>());
}

Json::Value make_event(const std::string& task_id, const std::string& type,
                       const std::string& rule_id, const std::string& rule_name,
                       uint32_t frame_index, int track_id = -1) {
    Json::Value event;
    event["id"] = task_id + ":" + rule_id + ":" + type + ":" +
                  std::to_string(frame_index) + ":" + std::to_string(track_id);
    event["type"]        = type;
    event["rule_id"]     = rule_id;
    event["rule_name"]   = rule_name;
    event["frame_index"] = frame_index;
    if (track_id >= 0) {
        event["track_id"] = track_id;
    }
    return event;
}

}  // namespace

AnalyticsNode::AnalyticsNode(const std::string& name) : GraphCore::Node(name) {
    m_type = GraphCore::MID_NODE;
}

bool AnalyticsNode::Init(YAML::Node config) {
    CHECK(GraphCore::Node::Init(config), "AnalyticsNode base initialization failed");
    CHECK(config["primary_instance"], "AnalyticsNode requires primary_instance");
    primary_instance_ = config["primary_instance"].as<std::string>();
    task_id_ = config["task_id"] ? config["task_id"].as<std::string>() : "";

    if (config["areas"]) {
        CHECK(config["areas"].IsSequence(), "AnalyticsNode areas must be a sequence");
        for (const auto& item : config["areas"]) {
            AreaRule rule;
            CHECK(item["id"] && item["polygon"], "AnalyticsNode area requires id and polygon");
            rule.id          = item["id"].as<std::string>();
            rule.name        = item["name"] ? item["name"].as<std::string>() : rule.id;
            rule.class_ids   = read_class_ids(item["classIds"]);
            rule.min_count   = item["minCount"] ? item["minCount"].as<int>() : 1;
            rule.hold_frames = item["holdFrames"] ? item["holdFrames"].as<int>() : 1;
            CHECK(item["polygon"].IsSequence() && item["polygon"].size() >= 3,
                  "AnalyticsNode area polygon requires at least three points");
            for (const auto& point : item["polygon"]) {
                rule.polygon.push_back(read_point(point));
            }
            areas_.push_back(std::move(rule));
        }
    }
    if (config["lines"]) {
        CHECK(config["lines"].IsSequence(), "AnalyticsNode lines must be a sequence");
        for (const auto& item : config["lines"]) {
            LineRule rule;
            CHECK(item["id"] && item["start"] && item["end"],
                  "AnalyticsNode line requires id, start and end");
            rule.id        = item["id"].as<std::string>();
            rule.name      = item["name"] ? item["name"].as<std::string>() : rule.id;
            rule.start     = read_point(item["start"]);
            rule.end       = read_point(item["end"]);
            rule.direction = item["direction"] ? item["direction"].as<std::string>() : "both";
            rule.class_ids = read_class_ids(item["classIds"]);
            CHECK(rule.direction == "both" || rule.direction == "a_to_b" ||
                      rule.direction == "b_to_a",
                  "AnalyticsNode line direction is invalid");
            lines_.push_back(std::move(rule));
        }
    }
    CHECK(!areas_.empty() || !lines_.empty(), "AnalyticsNode requires at least one area or line");
    return true;
}

bool AnalyticsNode::class_enabled(const std::unordered_set<int>& class_ids, int class_id) {
    return class_ids.empty() || class_ids.count(class_id) > 0;
}

float AnalyticsNode::line_side(const LineRule& line, const cv::Point2f& point) {
    return (line.end.x - line.start.x) * (point.y - line.start.y) -
           (line.end.y - line.start.y) * (point.x - line.start.x);
}

bool AnalyticsNode::crosses_segment(const LineRule& line, const cv::Point2f& before,
                                    const cv::Point2f& after) {
    const cv::Point2f movement = after - before;
    const cv::Point2f rule     = line.end - line.start;
    const float determinant    = movement.x * rule.y - movement.y * rule.x;
    if (std::abs(determinant) <= side_epsilon) {
        return false;
    }
    const cv::Point2f offset = line.start - before;
    const float t = (offset.x * rule.y - offset.y * rule.x) / determinant;
    const float u = (offset.x * movement.y - offset.y * movement.x) / determinant;
    return t >= 0.0F && t <= 1.0F && u >= 0.0F && u <= 1.0F;
}

void AnalyticsNode::cleanup_stale_tracks(AreaRule& rule, uint32_t frame_index) {
    for (auto item = rule.tracks.begin(); item != rule.tracks.end();) {
        if (frame_index - item->second.last_seen_frame > stale_track_frames) {
            item = rule.tracks.erase(item);
        } else {
            ++item;
        }
    }
}

void AnalyticsNode::cleanup_stale_tracks(LineRule& rule, uint32_t frame_index) {
    for (auto item = rule.tracks.begin(); item != rule.tracks.end();) {
        if (frame_index - item->second.last_seen_frame > stale_track_frames) {
            item = rule.tracks.erase(item);
        } else {
            ++item;
        }
    }
}

Data::BaseData::ptr AnalyticsNode::handle_data(Data::BaseData::ptr data) {
    return evaluate(std::move(data));
}

Data::BaseData::ptr AnalyticsNode::evaluate(Data::BaseData::ptr data) {
    const auto frame = data->get_frame_meta();
    if (!frame || frame->width == 0 || frame->height == 0 ||
        !data->has_frame_target_list(primary_instance_)) {
        return data;
    }
    const auto targets = data->get_frame_target_list(primary_instance_);
    Json::Value result;
    result["task_id"]     = task_id_;
    result["frame_index"] = frame->frame_index;
    Json::Value events(Json::arrayValue);
    Json::Value area_results(Json::arrayValue);
    Json::Value line_results(Json::arrayValue);

    for (auto& area : areas_) {
        int count = 0;
        for (size_t target_index = 0; target_index < targets->targets.size(); ++target_index) {
            const auto box = std::dynamic_pointer_cast<object_meta::FrameBoxTarget>(
                targets->targets[target_index]);
            if (!box || !class_enabled(area.class_ids, box->class_id)) {
                continue;
            }
            const cv::Point2f point(
                ((box->left + box->right) * 0.5F) / static_cast<float>(frame->width),
                box->bottom / static_cast<float>(frame->height));
            const bool inside = cv::pointPolygonTest(area.polygon, point, false) >= 0.0;
            if (inside) {
                ++count;
            }
            if (box->track_id <= 0) {
                continue;
            }
            auto& state = area.tracks[box->track_id];
            state.last_seen_frame = frame->frame_index;
            if (!state.initialized) {
                state.initialized    = true;
                state.pending_inside = inside;
                state.pending_frames = 1;
            } else if (inside == state.stable_inside) {
                state.pending_inside = inside;
                state.pending_frames = 0;
            } else if (inside == state.pending_inside) {
                ++state.pending_frames;
            } else {
                state.pending_inside = inside;
                state.pending_frames = 1;
            }
            if (state.pending_frames >= area.hold_frames &&
                state.stable_inside != state.pending_inside) {
                state.stable_inside = state.pending_inside;
                state.pending_frames = 0;
                auto event = make_event(task_id_, inside ? "area_enter" : "area_exit",
                                        area.id, area.name, frame->frame_index, box->track_id);
                event["class_id"]       = box->class_id;
                event["detection_index"] = static_cast<Json::UInt64>(target_index);
                events.append(std::move(event));
            }
        }
        const bool threshold_active = count >= area.min_count;
        if (threshold_active != area.threshold_active) {
            auto event = make_event(task_id_, threshold_active ? "area_threshold" :
                                    "area_threshold_cleared", area.id, area.name,
                                    frame->frame_index);
            event["count"]     = count;
            event["min_count"] = area.min_count;
            events.append(std::move(event));
            area.threshold_active = threshold_active;
        }
        cleanup_stale_tracks(area, frame->frame_index);
        Json::Value area_result;
        area_result["id"]        = area.id;
        area_result["name"]      = area.name;
        area_result["count"]     = count;
        area_result["min_count"] = area.min_count;
        area_result["active"]    = threshold_active;
        area_results.append(std::move(area_result));
    }

    for (auto& line : lines_) {
        for (size_t target_index = 0; target_index < targets->targets.size(); ++target_index) {
            const auto box = std::dynamic_pointer_cast<object_meta::FrameBoxTarget>(
                targets->targets[target_index]);
            if (!box || box->track_id <= 0 || !class_enabled(line.class_ids, box->class_id)) {
                continue;
            }
            const cv::Point2f point(
                ((box->left + box->right) * 0.5F) / static_cast<float>(frame->width),
                box->bottom / static_cast<float>(frame->height));
            const float side = line_side(line, point);
            auto& state = line.tracks[box->track_id];
            if (state.initialized && std::abs(state.side) > side_epsilon &&
                std::abs(side) > side_epsilon && state.side * side < 0.0F &&
                crosses_segment(line, state.point, point)) {
                const bool a_to_b = state.side < 0.0F && side > 0.0F;
                const std::string direction = a_to_b ? "a_to_b" : "b_to_a";
                const bool direction_enabled = line.direction == "both" || line.direction == direction;
                auto& seen = a_to_b ? line.a_to_b_tracks : line.b_to_a_tracks;
                if (direction_enabled && seen.insert(box->track_id).second) {
                    if (a_to_b) {
                        ++line.a_to_b_count;
                    } else {
                        ++line.b_to_a_count;
                    }
                    auto event = make_event(task_id_, "line_cross", line.id, line.name,
                                            frame->frame_index, box->track_id);
                    event["direction"]       = direction;
                    event["class_id"]        = box->class_id;
                    event["detection_index"] = static_cast<Json::UInt64>(target_index);
                    events.append(std::move(event));
                }
            }
            state.initialized     = true;
            state.point           = point;
            state.side            = side;
            state.last_seen_frame = frame->frame_index;
        }
        cleanup_stale_tracks(line, frame->frame_index);
        Json::Value line_result;
        line_result["id"]           = line.id;
        line_result["name"]         = line.name;
        line_result["a_to_b_count"] = Json::UInt64(line.a_to_b_count);
        line_result["b_to_a_count"] = Json::UInt64(line.b_to_a_count);
        line_result["total_count"]  = Json::UInt64(line.a_to_b_count + line.b_to_a_count);
        line_results.append(std::move(line_result));
    }
    result["areas"]  = std::move(area_results);
    result["lines"]  = std::move(line_results);
    result["events"] = std::move(events);
    data->set_analytics_result(std::move(result));
    return data;
}

}  // namespace Node

namespace {
Register<GraphCore::Node, Node::AnalyticsNode, std::string> register_analytics("AnalyticsNode");
}
