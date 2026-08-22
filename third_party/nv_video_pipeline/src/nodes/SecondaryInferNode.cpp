#include "SecondaryInferNode.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>

#include "BaseData.h"
#include "CLog.h"
#include "FrameBoxTarget.h"
#include "FrameMeta.h"
#include "InstancesManager.h"
#include "Register.h"
#include "StatusCode.h"

namespace Node {

SecondaryInferNode::SecondaryInferNode(const std::string& name) : GraphCore::Node(name) {
    m_type = GraphCore::MID_NODE;
}

bool SecondaryInferNode::Init(YAML::Node config) {
    CHECK(GraphCore::Node::Init(config), "SecondaryInferNode base initialization failed");
    CHECK(config["instance"] && config["primary_instance"],
          "SecondaryInferNode requires instance and primary_instance");
    instance_name_  = config["instance"].as<std::string>();
    primary_instance_ = config["primary_instance"].as<std::string>();
    CHECK(InstancesManager::get()->has_key(instance_name_),
          "SecondaryInferNode could not find its inference instance");
    instance_ = InstancesManager::get()->get_instance(instance_name_);
    if (config["source_class_ids"]) {
        CHECK(config["source_class_ids"].IsSequence(),
              "SecondaryInferNode source_class_ids must be a sequence");
        for (const auto& item : config["source_class_ids"]) {
            source_class_ids_.insert(item.as<int>());
        }
    }
    if (config["confidence_threshold"]) {
        confidence_threshold_ = config["confidence_threshold"].as<float>();
    }
    CHECK(confidence_threshold_ >= 0.0F && confidence_threshold_ <= 1.0F,
          "SecondaryInferNode confidence threshold is out of range");
    return true;
}

Data::BaseData::ptr SecondaryInferNode::handle_data(Data::BaseData::ptr data) {
    const auto frame = data->get_frame_meta();
    if (!frame || frame->frame.empty() || !data->has_frame_target_list(primary_instance_)) {
        return data;
    }
    const auto primary_targets = data->get_frame_target_list(primary_instance_);
    auto secondary_targets = std::make_shared<object_meta::FrameTargetList>();
    for (size_t index = 0; index < primary_targets->targets.size(); ++index) {
        const auto primary = std::dynamic_pointer_cast<object_meta::FrameBoxTarget>(
            primary_targets->targets[index]);
        if (!primary || (!source_class_ids_.empty() &&
                         source_class_ids_.count(primary->class_id) == 0)) {
            continue;
        }
        const int left   = std::clamp(static_cast<int>(std::floor(primary->left)), 0,
                                      std::max(0, frame->frame.cols - 1));
        const int top    = std::clamp(static_cast<int>(std::floor(primary->top)), 0,
                                      std::max(0, frame->frame.rows - 1));
        const int right  = std::clamp(static_cast<int>(std::ceil(primary->right)), left + 1,
                                      frame->frame.cols);
        const int bottom = std::clamp(static_cast<int>(std::ceil(primary->bottom)), top + 1,
                                      frame->frame.rows);
        if (right - left < 2 || bottom - top < 2) {
            continue;
        }
        auto child = std::make_shared<Data::BaseData>();
        child->data_name = data->data_name;
        child->set_frame_meta(std::make_shared<object_meta::FrameMeta>(
            frame->frame(cv::Rect(left, top, right - left, bottom - top)).clone(),
            frame->frame_index));
        Job job;
        job.data = child;
        if (!instance_->commit(job)) {
            LOG_WARN("SecondaryInferNode {} could not queue crop {}", getName(), index);
            continue;
        }
        auto future = job.promise->get_future();
        if (!future_wait_for_true(future, 5000) ||
            !child->has_frame_target_list(instance_name_)) {
            LOG_WARN("SecondaryInferNode {} timed out on crop {}", getName(), index);
            continue;
        }
        const auto child_targets = child->get_frame_target_list(instance_name_);
        for (const auto& target : child_targets->targets) {
            const auto box = std::dynamic_pointer_cast<object_meta::FrameBoxTarget>(target);
            if (!box || box->confidence < confidence_threshold_) {
                continue;
            }
            auto mapped = std::make_shared<object_meta::FrameBoxTarget>(*box);
            mapped->left   += static_cast<float>(left);
            mapped->right  += static_cast<float>(left);
            mapped->top    += static_cast<float>(top);
            mapped->bottom += static_cast<float>(top);
            mapped->parent_track_id = primary->track_id;
            mapped->parent_detection_index = static_cast<int>(index);
            mapped->parent_instance = primary_instance_;
            secondary_targets->targets.push_back(std::move(mapped));
        }
    }
    data->set_frame_target_list(instance_name_, std::move(secondary_targets));
    return data;
}

}  // namespace Node

namespace {
Register<GraphCore::Node, Node::SecondaryInferNode, std::string>
    register_secondary_infer("SecondaryInferNode");
}
