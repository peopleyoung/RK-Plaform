#pragma once

#include <cstdint>
#include <string>

#include <json/value.h>

#include "BaseData.h"

namespace Node {

Json::Value build_inference_payload(const Data::BaseData::ptr& data, const std::string& task_id = "",
                                    uint64_t revision = 0, const std::string& primary_instance = "");
std::string build_anhuan_message(const Data::BaseData::ptr& data, const std::string& task_id = "",
                                 uint64_t revision = 0, const std::string& primary_instance = "");
std::string default_anhuan_key(const std::string& input_uri);

}  // namespace Node
