#include "FrameBoxTarget.h"

#include <json/value.h>

#include <cmath>
#include <memory>

namespace object_meta {

Json::Value FrameBoxTarget::to_json() {
    Json::Value json_obj;
    json_obj["x"]          = static_cast<int>(std::round(left));
    json_obj["y"]          = static_cast<int>(std::round(top));
    json_obj["w"]          = static_cast<int>(std::round(right - left));
    json_obj["h"]          = static_cast<int>(std::round(bottom - top));
    json_obj["label"]      = label;
    json_obj["confidence"] = confidence;
    json_obj["class_id"]   = class_id;
    json_obj["track_id"]   = track_id;
    if (parent_track_id >= 0) {
        json_obj["parent_track_id"] = parent_track_id;
    }
    if (parent_detection_index >= 0) {
        json_obj["parent_detection_index"] = parent_detection_index;
    }
    if (!parent_instance.empty()) {
        json_obj["parent_instance"] = parent_instance;
    }

    return json_obj;
}

FrameTarget::ptr FrameBoxTarget::deep_copy() {
    return std::make_shared<FrameBoxTarget>(*this);
}
};  // namespace object_meta
