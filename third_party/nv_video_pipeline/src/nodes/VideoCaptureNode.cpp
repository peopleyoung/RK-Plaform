#include "VideoCaptureNode.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "BaseData.h"
#include "CLog.h"
#include "FrameMeta.h"
#include "PacketMeta.h"
#include "Register.h"
#include "StatusCode.h"

namespace Node {

VideoCaptureNode::VideoCaptureNode(const std::string& name) : GraphCore::Node(name) {
    m_type = GraphCore::SRC_NODE;
    set_after_start_cb([this]() {
        return after_start();
    });
    set_exit_cb([this]() {
        return on_exit();
    });
}

VideoCaptureNode::~VideoCaptureNode() {
    on_exit();
}

bool VideoCaptureNode::Init(YAML::Node config) {
    CHECK(GraphCore::Node::Init(config), "VideoCaptureNode base initialization failed");
    CHECK(config["input"], "VideoCaptureNode requires input");
    input_ = config["input"].as<std::string>();
    if (config["loop"]) {
        loop_ = config["loop"].as<bool>();
    }
    if (config["reconnect_ms"]) {
        reconnect_ms_ = std::max(0, config["reconnect_ms"].as<int>());
    }
    return open_capture();
}

bool VideoCaptureNode::open_capture() {
    capture_.release();
    single_image_.release();
    if (input_.rfind("rtsp://", 0) != 0) {
        const auto extension = std::filesystem::path(input_).extension().string();
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
            single_image_ = cv::imread(input_, cv::IMREAD_COLOR);
            if (single_image_.empty()) {
                LOG_ERROR("VideoCaptureNode {} could not read image {}", getName(), input_);
                return false;
            }
            fps_ = 1.0;
            LOG_INFO("VideoCaptureNode {} opened image {}", getName(), input_);
            return true;
        }
    }
    const std::vector<int> parameters = {
        cv::CAP_PROP_OPEN_TIMEOUT_MSEC,
        5000,
        cv::CAP_PROP_READ_TIMEOUT_MSEC,
        5000,
    };
    if (!capture_.open(input_, cv::CAP_FFMPEG, parameters)) {
        LOG_ERROR("VideoCaptureNode {} could not open {}", getName(), input_);
        return false;
    }
    fps_ = capture_.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps_) || fps_ <= 0.0) {
        fps_ = 25.0;
    }
    LOG_INFO("VideoCaptureNode {} opened {} at {:.2f} fps", getName(), input_, fps_);
    return true;
}

int VideoCaptureNode::after_start() {
    capture_running_ = true;
    capture_thread_  = std::thread(&VideoCaptureNode::capture_loop, this);
    return 0;
}

int VideoCaptureNode::on_exit() {
    capture_running_ = false;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    capture_.release();
    return 0;
}

void VideoCaptureNode::capture_loop() {
    while (capture_running_) {
        cv::Mat frame;
        if (!single_image_.empty()) {
            frame = single_image_.clone();
        } else if (!capture_.read(frame) || frame.empty()) {
            if (!loop_ && input_.rfind("rtsp://", 0) != 0) {
                LOG_INFO("VideoCaptureNode {} reached end of input", getName());
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_ms_));
            if (!open_capture()) {
                continue;
            }
            continue;
        }

        auto data       = std::make_shared<Data::BaseData>();
        data->data_name = getName();
        data->set_frame_meta(std::make_shared<object_meta::FrameMeta>(frame, frame_index_++));
        data->set_packet_meta(std::make_shared<object_meta::PacketMeta>(static_cast<int>(std::round(fps_)), frame.cols,
                                                                        frame.rows, 0, 0));
        add_data(data);
        if (!single_image_.empty()) {
            if (!loop_) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

Data::BaseData::ptr VideoCaptureNode::handle_data(Data::BaseData::ptr data) {
    return data;
}

}  // namespace Node

namespace {
Register<GraphCore::Node, Node::VideoCaptureNode, std::string> register_video_capture("VideoCaptureNode");
}
