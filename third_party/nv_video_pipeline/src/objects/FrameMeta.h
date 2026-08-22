#pragma once

#include <cstdint>
#include <memory>
#include <opencv2/core/mat.hpp>

#include "CLog.h"

namespace object_meta {
class FrameMeta {
public:
    using ptr = std::shared_ptr<FrameMeta>;

    FrameMeta(cv::Mat frame, uint32_t frame_index = 0) : frame(frame), frame_index(frame_index) {
        width  = frame.cols;
        height = frame.rows;
        if (width == 0 || height == 0) {
            LOG_ERROR("FrameMeta: frame size is 0");
        }
    }

    cv::Mat  frame;
    uint32_t frame_index;
    uint32_t width;
    uint32_t height;
};

}  // namespace object_meta
