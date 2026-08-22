#pragma once

#include <memory>
#include <string>
#include <utility>

#include "FrameTarget.h"

namespace object_meta {
class FrameBoxTarget : public FrameTarget {
public:
    using ptr = std::shared_ptr<FrameBoxTarget>;

    FrameBoxTarget(float left, float top, float right, float bottom, float confidence, int class_id, std::string label)
        : left(left),
          top(top),
          right(right),
          bottom(bottom),
          class_id(class_id),
          confidence(confidence),
          label(std::move(label)) {
    }

    ~FrameBoxTarget() override = default;

    Json::Value      to_json() override;
    FrameTarget::ptr deep_copy() override;

    float       left;
    float       top;
    float       right;
    float       bottom;
    int         class_id;
    float       confidence;
    std::string label;
    int         track_id = -1;
    int         parent_track_id = -1;
    int         parent_detection_index = -1;
    std::string parent_instance;
};

}  // namespace object_meta
