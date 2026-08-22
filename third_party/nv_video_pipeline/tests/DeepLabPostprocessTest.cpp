#include "DeepLabPostprocess.h"

#include <cassert>
#include <cstddef>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

cv::Mat reference_mask(const std::vector<float>& nchw, size_t classes, size_t height,
                       size_t width, int source_width, int source_height) {
    std::vector<cv::Mat> planes;
    for (size_t channel = 0; channel < classes; ++channel) {
        cv::Mat source(static_cast<int>(height), static_cast<int>(width), CV_32F,
                       const_cast<float*>(nchw.data() + channel * height * width));
        cv::Mat resized;
        cv::resize(source, resized, cv::Size(source_width, source_height), 0.0, 0.0,
                   cv::INTER_LINEAR);
        planes.push_back(std::move(resized));
    }
    cv::Mat result(source_height, source_width, CV_32S);
    for (int row = 0; row < source_height; ++row) {
        auto* output = result.ptr<int>(row);
        for (int column = 0; column < source_width; ++column) {
            size_t best = 0;
            float score = planes[0].at<float>(row, column);
            for (size_t channel = 1; channel < classes; ++channel) {
                const float candidate = planes[channel].at<float>(row, column);
                if (candidate > score) {
                    score = candidate;
                    best = channel;
                }
            }
            output[column] = static_cast<int>(best);
        }
    }
    return result;
}

void assert_equal(const cv::Mat& actual, const cv::Mat& expected) {
    assert(actual.type() == CV_32S);
    assert(actual.size() == expected.size());
    assert(cv::countNonZero(actual != expected) == 0);
}

}  // namespace

int main() {
    constexpr size_t classes = 3;
    constexpr size_t height = 2;
    constexpr size_t width = 2;
    const std::vector<float> nchw = {
        4.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 4.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 3.0F, 5.0F,
    };
    const cv::Mat expected = reference_mask(nchw, classes, height, width, 5, 3);

    cv::Mat nchw_mask;
    assert(deeplab_logits_to_mask(nchw.data(), classes, height, width, false, 5, 3,
                                  nchw_mask));
    assert_equal(nchw_mask, expected);

    std::vector<float> nhwc(classes * height * width);
    for (size_t row = 0; row < height; ++row) {
        for (size_t column = 0; column < width; ++column) {
            for (size_t channel = 0; channel < classes; ++channel) {
                nhwc[(row * width + column) * classes + channel] =
                    nchw[(channel * height + row) * width + column];
            }
        }
    }
    cv::Mat nhwc_mask;
    assert(deeplab_logits_to_mask(nhwc.data(), classes, height, width, true, 5, 3,
                                  nhwc_mask));
    assert_equal(nhwc_mask, expected);

    const std::vector<float> ties(classes * height * width, 1.0F);
    cv::Mat tie_mask;
    assert(deeplab_logits_to_mask(ties.data(), classes, height, width, false, 5, 3,
                                  tie_mask));
    assert(cv::countNonZero(tie_mask) == 0);

    cv::Mat rejected;
    assert(!deeplab_logits_to_mask(nullptr, classes, height, width, false, 5, 3,
                                   rejected));
    assert(!deeplab_logits_to_mask(nchw.data(), 0, height, width, false, 5, 3,
                                   rejected));
    assert(!deeplab_logits_to_mask(nchw.data(), classes, height, width, false, 0, 3,
                                   rejected));
    return 0;
}
