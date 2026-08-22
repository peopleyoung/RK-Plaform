#pragma once

#include <rknn_api.h>
#include <yaml-cpp/yaml.h>

#include <string>

#include "CLog.h"

namespace infer {

inline bool parse_rknn_core_config(const YAML::Node& config, rknn_core_mask& mask, std::string& mask_name,
                                   std::string& policy, const std::string& instance_name) {
    mask_name = config["core_mask"] ? config["core_mask"].as<std::string>() : "auto";
    policy    = config["core_policy"] ? config["core_policy"].as<std::string>() : "shared";
    if (policy != "shared" && policy != "exclusive") {
        LOG_ERROR("RKNN instance {} core_policy must be shared or exclusive, got {}", instance_name, policy);
        return false;
    }

    if (mask_name == "auto") {
        mask = RKNN_NPU_CORE_AUTO;
    } else if (mask_name == "core0") {
        mask = RKNN_NPU_CORE_0;
    } else if (mask_name == "core1") {
        mask = RKNN_NPU_CORE_1;
    } else if (mask_name == "core2") {
        mask = RKNN_NPU_CORE_2;
    } else if (mask_name == "core0_1") {
        mask = RKNN_NPU_CORE_0_1;
    } else if (mask_name == "core0_1_2") {
        mask = RKNN_NPU_CORE_0_1_2;
    } else {
        LOG_ERROR("RKNN instance {} has unsupported core_mask {}", instance_name, mask_name);
        return false;
    }

    if (policy == "exclusive" && mask_name == "auto") {
        LOG_ERROR("RKNN instance {} exclusive core_policy requires an explicit core_mask", instance_name);
        return false;
    }
    return true;
}

}  // namespace infer
