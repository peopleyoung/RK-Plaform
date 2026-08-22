#include "AnhuanMessage.h"

#include <json/json.h>

#include <sstream>
#include <vector>

namespace Node {

Json::Value build_inference_payload(const Data::BaseData::ptr& data, const std::string& task_id,
                                    uint64_t revision, const std::string& primary_instance) {
    Json::Value envelope;
    Json::Value detection_results(Json::objectValue);
    for (const auto& item : data->target_map) {
        Json::Value instance;
        instance["datas"]    = item.second ? item.second->to_json() : Json::Value(Json::objectValue);
        envelope[item.first] = std::move(instance);
        detection_results[item.first] = item.second ? item.second->to_json()["object"]
                                                    : Json::Value(Json::arrayValue);
    }
    Json::Value structured(Json::objectValue);
    for (const auto& item : data->inference_result_map) {
        if (!item.second) {
            continue;
        }
        Json::Value result;
        result["type"]   = item.second->type();
        result["result"] = item.second->to_json();
        structured[item.first] = std::move(result);
        Json::Value legacy_instance;
        legacy_instance["datas"] = item.second->to_json();
        envelope[item.first] = std::move(legacy_instance);
    }
    const auto frame = data->get_frame_meta();
    envelope["schema_version"] = 2;
    if (!task_id.empty()) {
        envelope["task_id"] = task_id;
    }
    if (revision > 0) {
        envelope["revision"] = Json::UInt64(revision);
    }
    if (!primary_instance.empty()) {
        envelope["primary_instance"] = primary_instance;
        envelope["instance"] = primary_instance;
    }
    const Json::Value frame_index = frame ? Json::Value(frame->frame_index) : Json::Value(0);
    envelope["index"]       = frame_index;
    envelope["frame_index"] = frame_index;
    if (frame) {
        envelope["width"]  = frame->width;
        envelope["height"] = frame->height;
    }
    envelope["create_time"] = Json::UInt64(data->create_time);
    envelope["detection_results"] = std::move(detection_results);
    if (!primary_instance.empty() && data->has_frame_target_list(primary_instance)) {
        envelope["detections"] = data->get_frame_target_list(primary_instance)->to_json()["object"];
    } else {
        envelope["detections"] = Json::Value(Json::arrayValue);
    }
    if (!primary_instance.empty() && data->has_frame_inference_result(primary_instance)) {
        const auto result = data->get_frame_inference_result(primary_instance);
        envelope["result_type"] = result->type();
        envelope["result"] = result->to_json();
    }
    envelope["structured_results"] = std::move(structured);
    envelope["analytics"] = data->get_analytics_result();
    envelope["media"] = data->get_media_result();
    return envelope;
}

std::string build_anhuan_message(const Data::BaseData::ptr& data, const std::string& task_id,
                                 uint64_t revision, const std::string& primary_instance) {
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(
        writer, build_inference_payload(data, task_id, revision, primary_instance));
}

std::string default_anhuan_key(const std::string& input_uri) {
    std::string path = input_uri;
    const auto query = path.find_first_of("?#");
    if (query != std::string::npos) {
        path.resize(query);
    }
    while (!path.empty() && path.back() == '/') {
        path.pop_back();
    }
    std::vector<std::string> segments;
    std::stringstream        stream(path);
    std::string              segment;
    while (std::getline(stream, segment, '/')) {
        if (!segment.empty()) {
            segments.push_back(segment);
        }
    }
    if (segments.size() >= 2) {
        return segments[segments.size() - 2] + "_" + segments.back();
    }
    return segments.empty() ? "inference" : segments.back();
}

}  // namespace Node
