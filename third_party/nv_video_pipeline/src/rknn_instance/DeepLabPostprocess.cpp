#include "DeepLabPostprocess.h"

#include <cstdint>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace {

constexpr size_t kMaxClasses = 4096;
constexpr size_t kMaxTensorDimension = 16384;
constexpr size_t kMaxTensorElements = 1024ULL * 1024ULL * 1024ULL;

bool valid_dimensions(size_t classes, size_t height, size_t width,
                      int source_width, int source_height) {
    if (classes == 0 || classes > kMaxClasses || height == 0 || width == 0
        || height > kMaxTensorDimension || width > kMaxTensorDimension
        || source_width <= 0 || source_height <= 0
        || static_cast<size_t>(source_width) > kMaxTensorDimension
        || static_cast<size_t>(source_height) > kMaxTensorDimension) {
        return false;
    }
    if (height > kMaxTensorElements / width) {
        return false;
    }
    const size_t plane_elements = height * width;
    return classes <= kMaxTensorElements / plane_elements;
}

void copy_class_plane(const float* logits, size_t classes, size_t height,
                      size_t width, bool nhwc, size_t channel, cv::Mat& plane) {
    plane.create(static_cast<int>(height), static_cast<int>(width), CV_32F);
    for (size_t row = 0; row < height; ++row) {
        float* output = plane.ptr<float>(static_cast<int>(row));
        for (size_t column = 0; column < width; ++column) {
            const size_t index = nhwc
                                     ? (row * width + column) * classes + channel
                                     : (channel * height + row) * width + column;
            output[column] = logits[index];
        }
    }
}

}  // namespace

bool deeplab_logits_to_mask(const float* logits, size_t classes, size_t height,
                            size_t width, bool nhwc, int source_width,
                            int source_height, cv::Mat& class_mask) {
    class_mask.release();
    if (logits == nullptr
        || !valid_dimensions(classes, height, width, source_width, source_height)) {
        return false;
    }

    cv::Mat plane;
    cv::Mat resized;
    cv::Mat best_scores;
    class_mask = cv::Mat::zeros(source_height, source_width, CV_32S);
    for (size_t channel = 0; channel < classes; ++channel) {
        copy_class_plane(logits, classes, height, width, nhwc, channel, plane);
        cv::resize(plane, resized, cv::Size(source_width, source_height), 0.0, 0.0,
                   cv::INTER_LINEAR);
        if (channel == 0) {
            resized.copyTo(best_scores);
            continue;
        }
        for (int row = 0; row < source_height; ++row) {
            const float* candidate = resized.ptr<float>(row);
            float* best = best_scores.ptr<float>(row);
            int32_t* classes_out = class_mask.ptr<int32_t>(row);
            for (int column = 0; column < source_width; ++column) {
                if (candidate[column] > best[column]) {
                    best[column] = candidate[column];
                    classes_out[column] = static_cast<int32_t>(channel);
                }
            }
        }
    }
    return true;
}
