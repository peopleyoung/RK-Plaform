#include "JsonOutputNode.h"

#include <json/json.h>

#include <cstdlib>
#include <filesystem>
#include <mutex>

#include "BaseData.h"
#include "AnhuanMessage.h"
#include "CLog.h"
#include "Register.h"
#include "StatusCode.h"

namespace Node {

namespace {
std::once_flag curl_init_once;
}

JsonOutputNode::JsonOutputNode(const std::string& name) : GraphCore::Node(name) {
    m_type = GraphCore::DES_NODE;
}

JsonOutputNode::~JsonOutputNode() {
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
}

bool JsonOutputNode::Init(YAML::Node config) {
    CHECK(GraphCore::Node::Init(config), "JsonOutputNode base initialization failed");
    CHECK(config["instance"], "JsonOutputNode requires instance");
    instance_name_ = config["instance"].as<std::string>();
    task_id_       = config["task_id"] ? config["task_id"].as<std::string>() : "";
    revision_      = config["revision"] ? config["revision"].as<uint64_t>() : 0;
    if (!config["output"]) {
        return true;
    }
    if (config["output"].IsScalar()) {
        const std::filesystem::path output_path(config["output"].as<std::string>());
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        output_.open(output_path, std::ios::out | std::ios::app);
        CHECK(output_, "JsonOutputNode could not open output file");
        return true;
    }
    CHECK(config["output"].IsMap(), "JsonOutputNode output must be a file path or mapping");
    const auto type = config["output"]["type"] ? config["output"]["type"].as<std::string>() : "jsonl";
    if (type == "jsonl") {
        CHECK(config["output"]["path"], "JsonOutputNode jsonl output requires path");
        const std::filesystem::path output_path(config["output"]["path"].as<std::string>());
        CHECK(!output_path.empty(), "JsonOutputNode jsonl path cannot be empty");
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        output_.open(output_path, std::ios::out | std::ios::app);
        CHECK(output_, "JsonOutputNode could not open output file");
        return true;
    }
    CHECK(type == "http", "JsonOutputNode output type must be jsonl or http");
    CHECK(config["output"]["url"], "JsonOutputNode http output requires url");
    http_url_ = config["output"]["url"].as<std::string>();
    CHECK(http_url_.rfind("http://", 0) == 0 || http_url_.rfind("https://", 0) == 0,
          "JsonOutputNode http url must use http or https");
    connect_timeout_ms_ =
        config["output"]["connect_timeout_ms"] ? config["output"]["connect_timeout_ms"].as<long>() : 1000;
    request_timeout_ms_ =
        config["output"]["request_timeout_ms"] ? config["output"]["request_timeout_ms"].as<long>() : 3000;
    CHECK(connect_timeout_ms_ >= 100 && connect_timeout_ms_ <= 60000,
          "JsonOutputNode connect timeout must be between 100 and 60000 ms");
    CHECK(request_timeout_ms_ >= connect_timeout_ms_ && request_timeout_ms_ <= 60000,
          "JsonOutputNode request timeout must cover connect timeout and be at most 60000 ms");
    if (config["output"]["authorization_env"]) {
        const auto variable = config["output"]["authorization_env"].as<std::string>();
        CHECK(!variable.empty(), "JsonOutputNode authorization environment variable cannot be empty");
        const char* token = std::getenv(variable.c_str());
        CHECK(token && token[0] != '\0', "JsonOutputNode authorization environment variable is unset");
        authorization_ = std::string("Authorization: Bearer ") + token;
    }
    std::call_once(curl_init_once, [] {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    });
    curl_ = curl_easy_init();
    CHECK(curl_, "JsonOutputNode could not initialize HTTP client");
    LOG_INFO("JsonOutputNode {} configured HTTP sink {}", getName(), http_url_);
    return true;
}

bool JsonOutputNode::emit(const std::string& line) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    // A default-constructed stream can still report goodbit; only an opened
    // file selects the JSONL sink. Otherwise structured results use HTTP.
    if (output_.is_open()) {
        output_ << line << '\n';
        output_.flush();
        return output_.good();
    }
    if (!curl_) {
        return true;
    }
    curl_slist* headers = nullptr;
    headers             = curl_slist_append(headers, "Content-Type: application/json");
    if (!authorization_.empty()) {
        headers = curl_slist_append(headers, authorization_.c_str());
    }
    curl_easy_setopt(curl_, CURLOPT_URL, http_url_.c_str());
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, line.data());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, static_cast<long>(line.size()));
    curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT_MS, connect_timeout_ms_);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT_MS, request_timeout_ms_);
    curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(
        curl_, CURLOPT_WRITEFUNCTION, +[](char*, size_t size, size_t count, void*) {
            return size * count;
        });
    const CURLcode result = curl_easy_perform(curl_);
    long           status = 0;
    if (result == CURLE_OK) {
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &status);
    }
    curl_slist_free_all(headers);
    if (result != CURLE_OK || status < 200 || status >= 300) {
        LOG_WARN("JsonOutputNode {} HTTP sink failed: curl={} status={}", getName(), curl_easy_strerror(result),
                 status);
        return false;
    }
    return true;
}

Data::BaseData::ptr JsonOutputNode::handle_data(Data::BaseData::ptr data) {
    if (!data->has_frame_target_list(instance_name_) &&
        !data->has_frame_inference_result(instance_name_)) {
        LOG_WARN("JsonOutputNode {} did not find inference result {}", getName(), instance_name_);
        return data;
    }
    const std::string line = build_anhuan_message(data, task_id_, revision_, instance_name_);
    if (!emit(line)) {
        LOG_WARN("JsonOutputNode {} could not deliver inference result", getName());
    }
    if (data->has_frame_target_list(instance_name_) &&
        data->get_frame_target_list(instance_name_)->size() == 0) {
        LOG_DEBUG("RKNN_RESULT {}", line);
    } else {
        LOG_INFO("RKNN_RESULT {}", line);
    }
    return data;
}

}  // namespace Node

namespace {
Register<GraphCore::Node, Node::JsonOutputNode, std::string> register_json_output("JsonOutputNode");
}
