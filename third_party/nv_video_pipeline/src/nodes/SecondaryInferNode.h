#pragma once

#include <memory>
#include <string>
#include <unordered_set>

#include "Instance.h"
#include "ProcessNode.h"

namespace Node {

class SecondaryInferNode : public GraphCore::Node {
public:
    explicit SecondaryInferNode(const std::string& name);
    bool Init(YAML::Node config) override;

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;

    infer::Instance::ptr   instance_;
    std::string            instance_name_;
    std::string            primary_instance_;
    std::unordered_set<int> source_class_ids_;
    float                  confidence_threshold_{0.25F};
};

}  // namespace Node
