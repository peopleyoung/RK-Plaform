#pragma once

#include <Util/util.h>

#include <algorithm>
#include <any>
#include <chrono>
#include <cstdlib>
#include <future>
#include <json/value.h>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "CLog.h"
#include "FrameInferenceResult.h"
#include "FrameMeta.h"
#include "FrameTarget.h"
#include "PacketMeta.h"

namespace Data {

// 用于存储数据的基类
class BaseData {
public:
    using ptr = std::shared_ptr<BaseData>;

    explicit BaseData() {
        create_time = toolkit::getCurrentMillisecond(true);
    }

    virtual ~BaseData() = default;

    uint64_t    create_time;  // 数据创建时间
    std::string data_name;    // 数据名称/数据来源

public:
    // 通过name插入任意类型的数据
    template <typename T>
    void Insert(const std::string &name, const T &value) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (data_map.find(name) == data_map.end()) {
            keys.push_back(name);
        }
        data_map[name] = value;
    }

    // 通过name删除数据
    void Remove(const std::string &name) {
        std::unique_lock<std::mutex> lock(m_mutex);
        data_map.erase(name);
        keys.erase(std::remove(keys.begin(), keys.end(), name), keys.end());
    }

    bool has_key(const std::string &name) {
        std::unique_lock<std::mutex> lock(m_mutex);
        return data_map.find(name) != data_map.end();
    }

    std::vector<std::string> get_all_keys() const {
        return keys;
    }

    template <typename T>
    T Pop(const std::string &name) {
        std::unique_lock<std::mutex> lock(m_mutex);
        for (auto it = data_map.begin(); it != data_map.end(); ++it) {
            if (it->first == name) {
                T value = std::any_cast<T>(it->second);
                data_map.erase(it);
                keys.erase(std::remove(keys.begin(), keys.end(), name), keys.end());
                return value;
            }
        }
        throw std::out_of_range("Key not found: " + name);
    }

    std::any get_data(const std::string &key) {
        std::unique_lock<std::mutex> lock(m_mutex);
        for (auto it = data_map.begin(); it != data_map.end(); ++it) {
            if (it->first == key) {
                return it->second;
            }
        }
        LOG_ERROR("Key not found: {}", key);
        return std::any();
    }

    std::mutex &get_mutex() {
        return m_mutex;
    }

    // 通过name获取任意类型的数据, 返回该类型数据的引用
    template <typename T>
    T &Get(const std::string &name) {
        std::unique_lock<std::mutex> lock(m_mutex);
        for (auto it = data_map.begin(); it != data_map.end(); ++it) {
            if (it->first == name) {
                return std::any_cast<T &>(it->second);
            }
        }
        LOG_ERROR("Key not found: {}", name);
        exit(EXIT_FAILURE);
        return std::any_cast<T &>(data_map[name]);
    }

    object_meta::FrameMeta::ptr get_frame_meta() {
        return m_frame_meta;
    }

    void set_frame_meta(object_meta::FrameMeta::ptr meta) {
        m_frame_meta = meta;
    }

    object_meta::PacketMeta::ptr get_packet_meta() {
        return m_packet_meta;
    }

    void set_packet_meta(object_meta::PacketMeta::ptr meta) {
        m_packet_meta = meta;
    }

    bool has_frame_target_list(const std::string &name) {
        return target_map.find(name) != target_map.end();
    }

    object_meta::FrameTargetList::ptr get_frame_target_list(const std::string &name) {
        if (!has_frame_target_list(name)) {
            LOG_ERROR("Frame target list not found: {}", name);
            return nullptr;
        }
        return target_map[name];
    }

    void set_frame_target_list(const std::string &name, object_meta::FrameTargetList::ptr list) {
        std::unique_lock<std::mutex> lock(m_mutex);
        target_map[name] = list;
    }

    bool has_frame_inference_result(const std::string &name) {
        std::unique_lock<std::mutex> lock(m_mutex);
        return inference_result_map.find(name) != inference_result_map.end();
    }

    object_meta::FrameInferenceResult::ptr get_frame_inference_result(const std::string &name) {
        std::unique_lock<std::mutex> lock(m_mutex);
        const auto                   item = inference_result_map.find(name);
        if (item == inference_result_map.end()) {
            LOG_ERROR("Frame inference result not found: {}", name);
            return nullptr;
        }
        return item->second;
    }

    void set_frame_inference_result(const std::string &name, object_meta::FrameInferenceResult::ptr result) {
        std::unique_lock<std::mutex> lock(m_mutex);
        inference_result_map[name] = std::move(result);
    }

    Json::Value get_analytics_result() {
        std::unique_lock<std::mutex> lock(m_mutex);
        return analytics_result_;
    }

    void set_analytics_result(Json::Value result) {
        std::unique_lock<std::mutex> lock(m_mutex);
        analytics_result_ = std::move(result);
    }

    Json::Value get_media_result() {
        std::unique_lock<std::mutex> lock(m_mutex);
        return media_result_;
    }

    void set_media_result(Json::Value result) {
        std::unique_lock<std::mutex> lock(m_mutex);
        media_result_ = std::move(result);
    }

public:
    std::unordered_map<std::string, object_meta::FrameTargetList::ptr>      target_map;
    std::unordered_map<std::string, object_meta::FrameInferenceResult::ptr> inference_result_map;

private:
    std::unordered_map<std::string, std::any> data_map;
    std::vector<std::string>                  keys;
    std::mutex                                m_mutex;
    object_meta::FrameMeta::ptr               m_frame_meta;
    object_meta::PacketMeta::ptr              m_packet_meta;
    Json::Value                               analytics_result_{Json::objectValue};
    Json::Value                               media_result_{Json::objectValue};
};

}  // namespace Data

struct Job {
    Data::BaseData::ptr                 data;
    std::shared_ptr<std::promise<bool>> promise = std::make_shared<std::promise<bool>>();
};

template <typename T>
bool future_wait_for(std::future<T> &future, T &result, int64_t timeout_ms = 1000) {
    auto status = future.wait_for(std::chrono::milliseconds(timeout_ms));
    if (status == std::future_status::timeout) {
        return false;
    }
    if (future.get() != result) {
        return false;
    }
    return true;
}

inline bool future_wait_for_true(std::future<bool> &future, int64_t timeout_ms = 1000) {
    bool result = true;
    return future_wait_for<bool>(future, result, timeout_ms);
}
