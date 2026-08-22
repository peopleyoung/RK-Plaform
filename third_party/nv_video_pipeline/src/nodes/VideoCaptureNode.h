#pragma once

#include <atomic>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>

#include "ProcessNode.h"

namespace Node {

class VideoCaptureNode : public GraphCore::Node {
public:
    explicit VideoCaptureNode(const std::string& name);
    ~VideoCaptureNode();

    bool Init(YAML::Node config) override;

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
    int                 after_start();
    int                 on_exit();
    void                capture_loop();
    bool                open_capture();

    std::string      input_;
    cv::VideoCapture capture_;
    cv::Mat          single_image_;
    std::thread      capture_thread_;
    std::atomic_bool capture_running_{false};
    bool             loop_         = false;
    int              reconnect_ms_ = 1000;
    int              frame_index_  = 0;
    double           fps_          = 0.0;
};

}  // namespace Node
