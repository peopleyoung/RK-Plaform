#include "RknnStructuredInstance.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <sstream>

#include "CLog.h"
#include "DeepLabPostprocess.h"
#include "Register.h"
#include "RknnCoreMask.h"

namespace infer {

namespace {

bool supported_type(const std::string& type) {
    return type == "DEEPLAB_LOGITS" || type == "PPOCR_DB" || type == "PPOCR_CTC";
}

}  // namespace

RknnStructuredInstance::RknnStructuredInstance(const std::string& name) : Instance(name) {
}

RknnStructuredInstance::~RknnStructuredInstance() {
    stop();
}

bool RknnStructuredInstance::init(YAML::Node config) {
    if (!config["model_path"] || !config["type"]) {
        LOG_ERROR("Structured RKNN instance {} requires model_path and type", getName());
        return false;
    }
    model_path_ = config["model_path"].as<std::string>();
    model_type_ = config["type"].as<std::string>();
    if (!parse_rknn_core_config(config, core_mask_, core_mask_name_, core_policy_, getName())) {
        return false;
    }
    if (!supported_type(model_type_)) {
        LOG_ERROR("Structured RKNN instance {} does not support type {}", getName(), model_type_);
        return false;
    }
    if (config["queue_capacity"]) {
        queue_capacity_ = std::max<size_t>(1, config["queue_capacity"].as<size_t>());
    }
    if (config["context_count"]) {
        const auto value = config["context_count"].as<long long>();
        if (value <= 0) {
            LOG_ERROR("Structured RKNN instance {} requires context_count > 0", getName());
            return false;
        }
        context_count_ = static_cast<size_t>(value);
    }
    if (config["worker_count"]) {
        const auto value = config["worker_count"].as<long long>();
        if (value <= 0) {
            LOG_ERROR("Structured RKNN instance {} requires worker_count > 0", getName());
            return false;
        }
        worker_count_ = static_cast<size_t>(value);
    }
    if (worker_count_ > context_count_) {
        LOG_ERROR("Structured RKNN instance {} requires worker_count <= context_count; workers={}, contexts={}",
                  getName(), worker_count_, context_count_);
        return false;
    }
    queue_capacity_ = std::max(queue_capacity_, worker_count_ * 2);
    if (config["binary_threshold"]) {
        binary_threshold_ = config["binary_threshold"].as<float>();
    }
    if (config["box_threshold"]) {
        box_threshold_ = config["box_threshold"].as<float>();
    }
    if (config["unclip_ratio"]) {
        unclip_ratio_ = config["unclip_ratio"].as<float>();
    }
    if (config["min_size"]) {
        min_size_ = config["min_size"].as<int>();
    }
    if (config["max_candidates"]) {
        const auto value = config["max_candidates"].as<long long>();
        if (value <= 0) {
            LOG_ERROR("Structured RKNN instance {} requires max_candidates > 0", getName());
            return false;
        }
        max_candidates_ = static_cast<size_t>(value);
    }
    if (config["max_regions"]) {
        const auto value = config["max_regions"].as<long long>();
        if (value <= 0) {
            LOG_ERROR("Structured RKNN instance {} requires max_regions > 0", getName());
            return false;
        }
        max_regions_ = static_cast<size_t>(value);
    }
    if (config["blank_index"]) {
        blank_index_ = config["blank_index"].as<int>();
    }
    if (config["ctc_scores_logits"]) {
        ctc_scores_logits_ = config["ctc_scores_logits"].as<bool>();
    }
    if (binary_threshold_ < 0.0f || binary_threshold_ > 1.0f || box_threshold_ < 0.0f || box_threshold_ > 1.0f
        || unclip_ratio_ < 1.0f || min_size_ < 1 || max_candidates_ == 0 || max_regions_ == 0) {
        LOG_ERROR("Structured RKNN instance {} has invalid post-processing thresholds", getName());
        return false;
    }

    const bool requires_labels = model_type_ == "DEEPLAB_LOGITS" || model_type_ == "PPOCR_CTC";
    if (requires_labels && !config["label_path"]) {
        LOG_ERROR("Structured RKNN instance {} requires label_path for {}", getName(), model_type_);
        return false;
    }
    if (config["label_path"]) {
        std::ifstream labels(config["label_path"].as<std::string>());
        for (std::string line; std::getline(labels, line);) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty()) {
                labels_.push_back(line);
            }
        }
    }
    if (requires_labels && labels_.empty()) {
        LOG_ERROR("Structured RKNN instance {} could not load labels", getName());
        return false;
    }
    std::vector<rknn_context> contexts;
    if (!load_model(model_path_, contexts)) {
        return false;
    }
    if (!query_tensors(contexts.front())) {
        for (const auto context : contexts) {
            rknn_destroy(context);
        }
        return false;
    }
    if (!pool_.init(std::move(contexts), worker_count_, queue_capacity_, [](rknn_context context) {
            rknn_destroy(context);
        })) {
        LOG_ERROR("Failed to initialize structured RKNN execution pool for {}", getName());
        return false;
    }
    LOG_INFO("Structured RKNN instance {} configured with {} contexts and {} workers", getName(), context_count_,
             worker_count_);
    return true;
}

bool RknnStructuredInstance::load_model(const std::string& path, std::vector<rknn_context>& contexts) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        LOG_ERROR("RKNN model not found: {}", path);
        return false;
    }
    const auto end = input.tellg();
    if (end <= 0) {
        LOG_ERROR("RKNN model is empty: {}", path);
        return false;
    }
    model_data_.resize(static_cast<size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(model_data_.data()), static_cast<std::streamsize>(model_data_.size()))) {
        LOG_ERROR("Failed to read RKNN model: {}", path);
        return false;
    }
    const bool created = RknnExecutionPool::create_contexts(
        context_count_,
        [this](rknn_context& context) {
            return rknn_init(&context, model_data_.data(), static_cast<uint32_t>(model_data_.size()), 0, nullptr);
        },
        [](rknn_context& source, rknn_context& context) { return rknn_dup_context(&source, &context); },
        [this](rknn_context context) { return rknn_set_core_mask(context, core_mask_); },
        [](rknn_context context) { rknn_destroy(context); }, contexts);
    if (!created) {
        LOG_ERROR("Failed to create {} structured RKNN contexts for {}", context_count_, path);
        return false;
    }
    LOG_INFO("Structured RKNN model {} uses NPU core mask {} ({})", path, core_mask_name_, core_policy_);
    return true;
}

bool RknnStructuredInstance::query_tensors(rknn_context context) {
    rknn_input_output_num io_count{};
    int                   status = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_count, sizeof(io_count));
    if (status != RKNN_SUCC || io_count.n_input != 1 || io_count.n_output != 1) {
        LOG_ERROR("Structured RKNN {} requires one input and one output; got status={}, inputs={}, outputs={}",
                  model_path_, status, io_count.n_input, io_count.n_output);
        return false;
    }
    input_attrs_.resize(1);
    output_attrs_.resize(1);
    input_attrs_[0]        = {};
    output_attrs_[0]       = {};
    input_attrs_[0].index  = 0;
    output_attrs_[0].index = 0;
    status                 = rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attrs_[0], sizeof(rknn_tensor_attr));
    if (status != RKNN_SUCC) {
        LOG_ERROR("Failed to query structured RKNN input {}", model_path_);
        return false;
    }
    status = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[0], sizeof(rknn_tensor_attr));
    if (status != RKNN_SUCC) {
        LOG_ERROR("Failed to query structured RKNN output {}", model_path_);
        return false;
    }
    LOG_INFO("Structured RKNN {} input={} output={} type={}", getName(), tensor_shape(input_attrs_[0]),
             tensor_shape(output_attrs_[0]), model_type_);
    const auto& input = input_attrs_[0];
    const bool  nchw  = input.fmt == RKNN_TENSOR_NCHW;
    if (input.n_dims != 4 || input.dims[0] != 1 || (input.fmt != RKNN_TENSOR_NCHW && input.fmt != RKNN_TENSOR_NHWC)
        || input.dims[nchw ? 1 : 3] != 3 || input.dims[nchw ? 2 : 1] == 0 || input.dims[nchw ? 3 : 2] == 0) {
        LOG_ERROR("Structured RKNN {} has unsupported input tensor {}", getName(), tensor_shape(input));
        return false;
    }
    const auto& output = output_attrs_[0];
    if (model_type_ == "DEEPLAB_LOGITS") {
        if (output.n_dims != 4 || output.dims[0] != 1
            || (output.fmt != RKNN_TENSOR_NCHW && output.fmt != RKNN_TENSOR_NHWC)
            || tensor_channels(output) != labels_.size()) {
            LOG_ERROR("DeepLab output {} does not expose {} class channels", tensor_shape(output), labels_.size());
            return false;
        }
    } else if (model_type_ == "PPOCR_DB") {
        if (output.n_dims != 4 || output.dims[0] != 1
            || (output.fmt != RKNN_TENSOR_NCHW && output.fmt != RKNN_TENSOR_NHWC) || tensor_channels(output) != 1) {
            LOG_ERROR("PPOCR DB output {} must be a rank-four single-channel map", tensor_shape(output));
            return false;
        }
    } else {
        const size_t classes    = labels_.size() + 1;
        bool         class_axis = false;
        for (uint32_t axis = output.n_dims == 3 ? 1 : 0; axis < output.n_dims; ++axis) {
            class_axis = class_axis || output.dims[axis] == classes;
        }
        if (output.n_dims < 2 || output.n_dims > 3 || (output.n_dims == 3 && output.dims[0] != 1) || !class_axis
            || blank_index_ < 0 || blank_index_ >= static_cast<int>(classes)) {
            LOG_ERROR("PPOCR CTC output {}, labels, or blank index {} are invalid", tensor_shape(output), blank_index_);
            return false;
        }
    }
    return true;
}

bool RknnStructuredInstance::commit(Job& job) {
    Job queued = job;
    return pool_.commit(
        [this, queued](rknn_context context) mutable {
            const auto begin = std::chrono::steady_clock::now();
            if (!process(queued, context)) {
                fail_job(queued, "structured RKNN inference failed");
                return;
            }
            {
                std::lock_guard<std::mutex> lock(perf_mutex_);
                perf_time_ms_ +=
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                perf_count_ += 1.0f;
            }
            try {
                queued.promise->set_value(true);
            } catch (const std::future_error&) {
            }
        },
        [this, queued]() mutable { fail_job(queued, "instance stopped before inference"); });
}

void RknnStructuredInstance::start() {
    pool_.start();
    LOG_INFO("Structured RKNN instance {} started", getName());
}

void RknnStructuredInstance::stop() {
    pool_.stop();
}

std::tuple<double, float> RknnStructuredInstance::get_perf() {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    const auto                  result = std::make_tuple(perf_time_ms_, perf_count_);
    perf_time_ms_                      = 0.0;
    perf_count_                        = 0.0f;
    return result;
}

bool RknnStructuredInstance::process(Job& job, rknn_context context) {
    if (!job.data || !job.data->get_frame_meta() || job.data->get_frame_meta()->frame.empty()) {
        LOG_ERROR("Structured RKNN instance {} received an empty frame", getName());
        return false;
    }
    const auto& input_attr = input_attrs_.front();
    const bool  nchw       = input_attr.fmt == RKNN_TENSOR_NCHW;
    const int   input_h    = static_cast<int>(input_attr.dims[nchw ? 2 : 1]);
    const int   input_w    = static_cast<int>(input_attr.dims[nchw ? 3 : 2]);
    const auto& source     = job.data->get_frame_meta()->frame;
    cv::Mat     resized;
    if (model_type_ == "PPOCR_CTC") {
        const float scale = input_h / static_cast<float>(source.rows);
        const int   width = std::clamp(static_cast<int>(std::round(source.cols * scale)), 1, input_w);
        cv::resize(source, resized, cv::Size(width, input_h));
        cv::Mat padded(input_h, input_w, CV_8UC3, cv::Scalar(0, 0, 0));
        resized.copyTo(padded(cv::Rect(0, 0, width, input_h)));
        resized = std::move(padded);
    } else {
        cv::resize(source, resized, cv::Size(input_w, input_h));
    }
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rknn_input input{};
    input.index        = 0;
    input.type         = RKNN_TENSOR_UINT8;
    input.fmt          = RKNN_TENSOR_NHWC;
    input.size         = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
    input.buf          = rgb.data;
    input.pass_through = 0;
    int status         = rknn_inputs_set(context, 1, &input);
    if (status == RKNN_SUCC) {
        status = rknn_run(context, nullptr);
    }
    if (status != RKNN_SUCC) {
        LOG_ERROR("Structured RKNN execution failed for {} with status {}", getName(), status);
        return false;
    }
    rknn_output output{};
    output.index      = 0;
    output.want_float = 1;
    status            = rknn_outputs_get(context, 1, &output, nullptr);
    if (status != RKNN_SUCC || output.buf == nullptr) {
        LOG_ERROR("Structured RKNN output acquisition failed for {} with status {}", getName(), status);
        return false;
    }
    const auto element_count = tensor_element_count(output_attrs_.front());
    const bool buffer_ok     = element_count > 0 && (output.size == 0 || output.size >= element_count * sizeof(float));
    object_meta::FrameInferenceResult::ptr result;
    bool                                   decoded = buffer_ok;
    if (decoded && model_type_ == "DEEPLAB_LOGITS") {
        decoded = decode_deeplab(static_cast<const float*>(output.buf), output_attrs_.front(),
                                 *job.data->get_frame_meta(), result);
    } else if (decoded && model_type_ == "PPOCR_DB") {
        decoded = decode_ppocr_db(static_cast<const float*>(output.buf), output_attrs_.front(),
                                  *job.data->get_frame_meta(), result);
    } else if (decoded) {
        decoded = decode_ppocr_ctc(static_cast<const float*>(output.buf), output_attrs_.front(), result);
    }
    rknn_outputs_release(context, 1, &output);
    if (!decoded || !result) {
        LOG_ERROR("Structured RKNN output decode failed for {}", getName());
        return false;
    }
    job.data->set_frame_inference_result(getName(), std::move(result));
    return true;
}

bool RknnStructuredInstance::decode_deeplab(const float* output, const rknn_tensor_attr& attr,
                                            const object_meta::FrameMeta&           frame,
                                            object_meta::FrameInferenceResult::ptr& result) {
    const size_t channels = tensor_channels(attr);
    const size_t height   = tensor_height(attr);
    const size_t width    = tensor_width(attr);
    if (!output || channels != labels_.size() || height == 0 || width == 0) {
        return false;
    }
    cv::Mat class_mask;
    if (!deeplab_logits_to_mask(output, channels, height, width,
                                attr.fmt == RKNN_TENSOR_NHWC, frame.width,
                                frame.height, class_mask)) {
        return false;
    }
    result = std::make_shared<object_meta::FrameSegmentationResult>(
        std::move(class_mask), labels_, frame.width, frame.height);
    return true;
}

bool RknnStructuredInstance::decode_ppocr_db(const float* output, const rknn_tensor_attr& attr,
                                             const object_meta::FrameMeta&           frame,
                                             object_meta::FrameInferenceResult::ptr& result) {
    const size_t height = tensor_height(attr);
    const size_t width  = tensor_width(attr);
    if (!output || height == 0 || width == 0 || tensor_channels(attr) != 1) {
        return false;
    }
    auto value = [&](size_t row, size_t column) {
        if (attr.fmt == RKNN_TENSOR_NHWC) {
            return output[row * width + column];
        }
        return output[row * width + column];
    };
    cv::Mat probability(static_cast<int>(height), static_cast<int>(width), CV_32F);
    for (size_t row = 0; row < height; ++row) {
        auto* values = probability.ptr<float>(static_cast<int>(row));
        for (size_t column = 0; column < width; ++column) {
            values[column] = value(row, column);
        }
    }
    cv::Mat binary;
    cv::threshold(probability, binary, binary_threshold_, 255.0, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8U);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(), [](const auto& lhs, const auto& rhs) {
        return std::abs(cv::contourArea(lhs)) > std::abs(cv::contourArea(rhs));
    });
    if (contours.size() > max_candidates_) {
        contours.resize(max_candidates_);
    }
    std::vector<object_meta::OcrRegion> regions;
    const float                         scale_x = frame.width / static_cast<float>(width);
    const float                         scale_y = frame.height / static_cast<float>(height);
    for (const auto& contour : contours) {
        if (contour.size() < 3) {
            continue;
        }
        const auto  rect       = cv::minAreaRect(contour);
        const float short_side = std::min(rect.size.width, rect.size.height);
        if (!std::isfinite(short_side) || short_side < min_size_) {
            continue;
        }
        cv::Mat                             region_mask = cv::Mat::zeros(probability.size(), CV_8U);
        std::vector<std::vector<cv::Point>> mask_contours{contour};
        cv::drawContours(region_mask, mask_contours, 0, cv::Scalar(1), cv::FILLED);
        const float score = static_cast<float>(cv::mean(probability, region_mask)[0]);
        if (!std::isfinite(score) || score < box_threshold_) {
            continue;
        }
        const double area      = std::abs(cv::contourArea(contour));
        const double perimeter = cv::arcLength(contour, true);
        if (!std::isfinite(area) || !std::isfinite(perimeter) || perimeter <= 0.0) {
            continue;
        }
        const float     distance = static_cast<float>(area * unclip_ratio_ / perimeter);
        cv::RotatedRect expanded(
            rect.center, cv::Size2f(rect.size.width + 2.0f * distance, rect.size.height + 2.0f * distance), rect.angle);
        cv::Point2f points[4];
        expanded.points(points);
        object_meta::OcrRegion region;
        region.confidence = score;
        for (const auto& point : points) {
            region.points.emplace_back(std::clamp(point.x * scale_x, 0.0f, static_cast<float>(frame.width - 1)),
                                       std::clamp(point.y * scale_y, 0.0f, static_cast<float>(frame.height - 1)));
        }
        regions.push_back(std::move(region));
    }
    std::sort(regions.begin(), regions.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.points.empty() || rhs.points.empty()) {
            return lhs.points.size() > rhs.points.size();
        }
        return lhs.points.front().y == rhs.points.front().y ? lhs.points.front().x < rhs.points.front().x
                                                            : lhs.points.front().y < rhs.points.front().y;
    });
    if (regions.size() > max_regions_) {
        regions.resize(max_regions_);
    }
    result = std::make_shared<object_meta::FrameOcrResult>("ocr_detection", std::move(regions));
    return true;
}

bool RknnStructuredInstance::decode_ppocr_ctc(const float* output, const rknn_tensor_attr& attr,
                                              object_meta::FrameInferenceResult::ptr& result) {
    if (!output || labels_.empty() || attr.n_dims < 2) {
        return false;
    }
    const size_t element_count = tensor_element_count(attr);
    const size_t classes       = labels_.size() + 1;
    int          class_axis    = -1;
    for (int axis = static_cast<int>(attr.n_dims) - 1; axis >= 0; --axis) {
        if (attr.dims[axis] == classes) {
            class_axis = axis;
            break;
        }
    }
    if (class_axis < 0 || element_count == 0 || element_count % classes != 0) {
        LOG_ERROR("PPOCR CTC output {} does not contain {} classes", tensor_shape(attr), classes);
        return false;
    }
    const size_t time_steps = element_count / classes;
    auto         value      = [&](size_t time, size_t class_index) {
        if (class_axis == static_cast<int>(attr.n_dims) - 1) {
            return output[time * classes + class_index];
        }
        return output[class_index * time_steps + time];
    };
    std::string text;
    float       confidence_sum   = 0.0f;
    size_t      confidence_count = 0;
    int         previous         = blank_index_;
    for (size_t time = 0; time < time_steps; ++time) {
        int   best_class = 0;
        float best_score = value(time, 0);
        for (size_t class_index = 1; class_index < classes; ++class_index) {
            const float score = value(time, class_index);
            if (score > best_score) {
                best_score = score;
                best_class = static_cast<int>(class_index);
            }
        }
        if (ctc_scores_logits_) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t class_index = 0; class_index < classes; ++class_index) {
                maximum = std::max(maximum, value(time, class_index));
            }
            float denominator = 0.0f;
            for (size_t class_index = 0; class_index < classes; ++class_index) {
                denominator += std::exp(value(time, class_index) - maximum);
            }
            best_score = denominator > 0.0f ? std::exp(best_score - maximum) / denominator : 0.0f;
        }
        if (best_class != blank_index_ && best_class != previous) {
            const int label_index = best_class > blank_index_ ? best_class - 1 : best_class;
            if (label_index >= 0 && label_index < static_cast<int>(labels_.size())) {
                text += labels_[label_index];
                confidence_sum += best_score;
                ++confidence_count;
            }
        }
        previous = best_class;
    }
    const float confidence = confidence_count > 0 ? confidence_sum / confidence_count : 0.0f;
    result = std::make_shared<object_meta::FrameOcrResult>("ocr_recognition", std::vector<object_meta::OcrRegion>{},
                                                           std::move(text), confidence);
    return true;
}

void RknnStructuredInstance::fail_job(Job& job, const std::string& reason) {
    LOG_ERROR("Structured RKNN instance {}: {}", getName(), reason);
    try {
        job.promise->set_value(false);
    } catch (const std::future_error&) {
    }
}

size_t RknnStructuredInstance::tensor_element_count(const rknn_tensor_attr& attr) {
    if (attr.n_elems != 0) {
        return attr.n_elems;
    }
    size_t count = 1;
    for (uint32_t index = 0; index < attr.n_dims; ++index) {
        if (attr.dims[index] == 0 || count > std::numeric_limits<size_t>::max() / attr.dims[index]) {
            return 0;
        }
        count *= attr.dims[index];
    }
    return count;
}

uint32_t RknnStructuredInstance::tensor_channels(const rknn_tensor_attr& attr) {
    if (attr.n_dims != 4) {
        return 0;
    }
    if (attr.fmt == RKNN_TENSOR_NCHW) {
        return attr.dims[1];
    }
    if (attr.fmt == RKNN_TENSOR_NHWC) {
        return attr.dims[3];
    }
    return attr.dims[1] <= attr.dims[3] ? attr.dims[1] : attr.dims[3];
}

uint32_t RknnStructuredInstance::tensor_height(const rknn_tensor_attr& attr) {
    if (attr.n_dims != 4) {
        return 0;
    }
    return attr.fmt == RKNN_TENSOR_NCHW ? attr.dims[2] : attr.dims[1];
}

uint32_t RknnStructuredInstance::tensor_width(const rknn_tensor_attr& attr) {
    if (attr.n_dims != 4) {
        return 0;
    }
    return attr.fmt == RKNN_TENSOR_NCHW ? attr.dims[3] : attr.dims[2];
}

std::string RknnStructuredInstance::tensor_shape(const rknn_tensor_attr& attr) {
    std::ostringstream stream;
    stream << '[';
    for (uint32_t index = 0; index < attr.n_dims; ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << attr.dims[index];
    }
    stream << ']';
    return stream.str();
}

}  // namespace infer

namespace {
Register<infer::Instance, infer::RknnStructuredInstance, std::string> register_rknn_structured("rknn_structured");
}
