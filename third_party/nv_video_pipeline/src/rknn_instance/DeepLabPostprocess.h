#pragma once

#include <cstddef>

#include <opencv2/core/mat.hpp>

bool deeplab_logits_to_mask(const float* logits, size_t classes, size_t height,
                            size_t width, bool nhwc, int source_width,
                            int source_height, cv::Mat& class_mask);
