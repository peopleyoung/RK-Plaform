#include "FrameInferenceResult.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace object_meta {

FrameSegmentationResult::FrameSegmentationResult(cv::Mat mask, std::vector<std::string> labels, int source_width,
                                                 int source_height)
    : mask_(std::move(mask)), labels_(std::move(labels)), source_width_(source_width), source_height_(source_height) {
    if (mask_.empty()) {
        return;
    }
    run_count_ = 1;
    int previous = mask_.at<int32_t>(0, 0);
    for (int row = 0; row < mask_.rows; ++row) {
        const auto* values = mask_.ptr<int32_t>(row);
        for (int column = 0; column < mask_.cols; ++column) {
            if (values[column] != previous) {
                ++run_count_;
                previous = values[column];
            }
        }
    }
}

std::string FrameSegmentationResult::type() const {
    return "segmentation";
}

Json::Value FrameSegmentationResult::to_json() const {
    Json::Value result;
    result["width"]         = mask_.cols;
    result["height"]        = mask_.rows;
    result["source_width"]  = source_width_;
    result["source_height"] = source_height_;
    result["encoding"]      = "class-rle-v1";
    Json::Value labels(Json::arrayValue);
    for (const auto& label : labels_) {
        labels.append(label);
    }
    result["labels"] = std::move(labels);

    Json::Value runs(Json::arrayValue);
    if (!mask_.empty()) {
        int current = mask_.at<int32_t>(0, 0);
        int count   = 0;
        for (int row = 0; row < mask_.rows; ++row) {
            const auto* values = mask_.ptr<int32_t>(row);
            for (int column = 0; column < mask_.cols; ++column) {
                const int value = values[column];
                if (value == current) {
                    ++count;
                    continue;
                }
                Json::Value run(Json::arrayValue);
                run.append(current);
                run.append(count);
                runs.append(std::move(run));
                current = value;
                count   = 1;
            }
        }
        Json::Value run(Json::arrayValue);
        run.append(current);
        run.append(count);
        runs.append(std::move(run));
    }
    result["runs"] = std::move(runs);
    return result;
}

FrameInferenceResult::ptr FrameSegmentationResult::deep_copy() const {
    return std::make_shared<FrameSegmentationResult>(mask_.clone(), labels_, source_width_, source_height_);
}

const cv::Mat& FrameSegmentationResult::mask() const {
    return mask_;
}

const std::vector<std::string>& FrameSegmentationResult::labels() const {
    return labels_;
}

int FrameSegmentationResult::source_width() const {
    return source_width_;
}

int FrameSegmentationResult::source_height() const {
    return source_height_;
}

size_t FrameSegmentationResult::run_count() const {
    return run_count_;
}

FrameOcrResult::FrameOcrResult(std::string result_type, std::vector<OcrRegion> regions, std::string text,
                               float confidence)
    : result_type_(std::move(result_type)),
      regions_(std::move(regions)),
      text_(std::move(text)),
      confidence_(confidence) {
}

std::string FrameOcrResult::type() const {
    return result_type_;
}

Json::Value FrameOcrResult::to_json() const {
    Json::Value result;
    if (result_type_ == "ocr_recognition") {
        result["text"]       = text_;
        result["confidence"] = confidence_;
        return result;
    }
    Json::Value regions(Json::arrayValue);
    for (const auto& region : regions_) {
        Json::Value item;
        item["confidence"] = region.confidence;
        if (!region.text.empty()) {
            item["text"] = region.text;
        }
        Json::Value points(Json::arrayValue);
        for (const auto& point : region.points) {
            Json::Value coordinates(Json::arrayValue);
            coordinates.append(point.x);
            coordinates.append(point.y);
            points.append(std::move(coordinates));
        }
        item["points"] = std::move(points);
        regions.append(std::move(item));
    }
    result["regions"] = std::move(regions);
    return result;
}

FrameInferenceResult::ptr FrameOcrResult::deep_copy() const {
    return std::make_shared<FrameOcrResult>(result_type_, regions_, text_, confidence_);
}

const std::vector<OcrRegion>& FrameOcrResult::regions() const {
    return regions_;
}

const std::string& FrameOcrResult::text() const {
    return text_;
}

float FrameOcrResult::confidence() const {
    return confidence_;
}

}  // namespace object_meta
