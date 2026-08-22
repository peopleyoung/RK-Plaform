#pragma once

#include <json/value.h>

#include <memory>
#include <string>
#include <vector>

namespace object_meta {

class FrameTarget {
public:
    using ptr              = std::shared_ptr<FrameTarget>;
    virtual ~FrameTarget() = default;

    virtual Json::Value      to_json()   = 0;
    virtual FrameTarget::ptr deep_copy() = 0;
};

class FrameTargetList {
public:
    using ptr         = std::shared_ptr<FrameTargetList>;
    FrameTargetList() = default;

    Json::Value to_json() {
        Json::Value json_array(Json::arrayValue);
        for (auto& target : targets) {
            json_array.append(target->to_json());
        }
        Json::Value data;
        data["object"] = json_array;
        return data;
    }

    FrameTargetList::ptr deep_copy() {
        auto list_copy = std::make_shared<FrameTargetList>();
        for (auto& target : targets) {
            list_copy->targets.push_back(target->deep_copy());
        }
        return list_copy;
    }

    int size() const {
        return targets.size();
    }

public:
    std::vector<FrameTarget::ptr> targets;
};

}  // namespace object_meta
