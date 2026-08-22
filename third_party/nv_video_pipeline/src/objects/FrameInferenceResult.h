#pragma once

#include <json/value.h>

#include <cstddef>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

namespace object_meta {

class FrameInferenceResult {
public:
    using ptr                       = std::shared_ptr<FrameInferenceResult>;
    virtual ~FrameInferenceResult() = default;

    virtual std::string               type() const      = 0;
    virtual Json::Value               to_json() const   = 0;
    virtual FrameInferenceResult::ptr deep_copy() const = 0;
};

class FrameSegmentationResult final : public FrameInferenceResult {
public:
    FrameSegmentationResult(cv::Mat mask, std::vector<std::string> labels, int source_width, int source_height);

    std::string               type() const override;
    Json::Value               to_json() const override;
    FrameInferenceResult::ptr deep_copy() const override;

    const cv::Mat&                  mask() const;
    const std::vector<std::string>& labels() const;
    int                             source_width() const;
    int                             source_height() const;
    size_t                          run_count() const;

private:
    cv::Mat                  mask_;
    std::vector<std::string> labels_;
    int                      source_width_  = 0;
    int                      source_height_ = 0;
    size_t                   run_count_ = 0;
};

struct OcrRegion {
    std::vector<cv::Point2f> points;
    float                    confidence = 0.0f;
    std::string              text;
};

class FrameOcrResult final : public FrameInferenceResult {
public:
    FrameOcrResult(std::string result_type, std::vector<OcrRegion> regions, std::string text = "",
                   float confidence = 0.0f);

    std::string               type() const override;
    Json::Value               to_json() const override;
    FrameInferenceResult::ptr deep_copy() const override;

    const std::vector<OcrRegion>& regions() const;
    const std::string&            text() const;
    float                         confidence() const;

private:
    std::string            result_type_;
    std::vector<OcrRegion> regions_;
    std::string            text_;
    float                  confidence_ = 0.0f;
};

}  // namespace object_meta
