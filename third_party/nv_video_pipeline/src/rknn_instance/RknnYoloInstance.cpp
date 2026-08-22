#include "RknnYoloInstance.h"

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
#include "FrameBoxTarget.h"
#include "Register.h"
#include "RknnCoreMask.h"
#include "StatusCode.h"

namespace {

std::string tensor_shape(const rknn_tensor_attr& attr) {
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

size_t tensor_element_count(const rknn_tensor_attr& attr) {
    if (attr.n_elems != 0) {
        return attr.n_elems;
    }
    if (attr.n_dims == 0) {
        return 0;
    }

    size_t elements = 1;
    for (uint32_t index = 0; index < attr.n_dims; ++index) {
        const size_t dimension = attr.dims[index];
        if (dimension == 0 || elements > std::numeric_limits<size_t>::max() / dimension) {
            return 0;
        }
        elements *= dimension;
    }
    return elements;
}

uint32_t tensor_channels(const rknn_tensor_attr& attr) {
    if (attr.n_dims != 4) {
        return 0;
    }
    return attr.fmt == RKNN_TENSOR_NHWC ? attr.dims[3] : attr.dims[1];
}

uint32_t tensor_height(const rknn_tensor_attr& attr) {
    if (attr.n_dims != 4) {
        return 0;
    }
    return attr.fmt == RKNN_TENSOR_NHWC ? attr.dims[1] : attr.dims[2];
}

uint32_t tensor_width(const rknn_tensor_attr& attr) {
    if (attr.n_dims != 4) {
        return 0;
    }
    return attr.fmt == RKNN_TENSOR_NHWC ? attr.dims[2] : attr.dims[3];
}

size_t detection_feature_count(const rknn_tensor_attr& attr) {
    if (attr.n_dims < 2) {
        return 0;
    }
    const size_t trailing = attr.dims[attr.n_dims - 1];
    if (trailing >= 6 && trailing <= 512) {
        return trailing;
    }
    const size_t preceding = attr.dims[attr.n_dims - 2];
    return preceding >= 6 && preceding <= 512 ? preceding : 0;
}

float intersection_over_union(const infer::RknnYoloInstance::Candidate& lhs,
                              const infer::RknnYoloInstance::Candidate& rhs) {
    const float left         = std::max(lhs.left, rhs.left);
    const float top          = std::max(lhs.top, rhs.top);
    const float right        = std::min(lhs.right, rhs.right);
    const float bottom       = std::min(lhs.bottom, rhs.bottom);
    const float intersection = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    const float lhs_area     = std::max(0.0f, lhs.right - lhs.left) * std::max(0.0f, lhs.bottom - lhs.top);
    const float rhs_area     = std::max(0.0f, rhs.right - rhs.left) * std::max(0.0f, rhs.bottom - rhs.top);
    const float denominator  = lhs_area + rhs_area - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

}  // namespace

namespace infer {

RknnYoloInstance::RknnYoloInstance(const std::string& name) : Instance(name) {
}

RknnYoloInstance::~RknnYoloInstance() {
    stop();
}

bool RknnYoloInstance::init(YAML::Node config) {
    CHECK(config["model_path"], "RKNN instance requires model_path");
    CHECK(config["label_path"], "RKNN instance requires label_path");
    CHECK(config["type"], "RKNN instance requires type");

    model_path_ = config["model_path"].as<std::string>();
    model_type_ = config["type"].as<std::string>();
    if (!parse_rknn_core_config(config, core_mask_, core_mask_name_, core_policy_, getName())) {
        return false;
    }
    if (model_type_ != "V5" && model_type_ != "ByteTrack" && model_type_ != "YOLO_DFL_SPLIT") {
        LOG_ERROR(
            "RKNN instance {} does not support model type {}; supported types are V5, ByteTrack and YOLO_DFL_SPLIT",
            getName(), model_type_);
        return false;
    }
    if (config["confidence_threshold"]) {
        confidence_threshold_ = config["confidence_threshold"].as<float>();
    }
    if (config["nms_threshold"]) {
        nms_threshold_ = config["nms_threshold"].as<float>();
    }
    if (config["class_scores_logits"]) {
        class_scores_logits_ = config["class_scores_logits"].as<bool>();
    }
    if (config["context_count"]) {
        const auto value = config["context_count"].as<long long>();
        if (value <= 0) {
            LOG_ERROR("RKNN instance {} requires context_count > 0", getName());
            return false;
        }
        context_count_ = static_cast<size_t>(value);
    }
    if (config["worker_count"]) {
        const auto value = config["worker_count"].as<long long>();
        if (value <= 0) {
            LOG_ERROR("RKNN instance {} requires worker_count > 0", getName());
            return false;
        }
        worker_count_ = static_cast<size_t>(value);
    }
    if (worker_count_ > context_count_) {
        LOG_ERROR("RKNN instance {} requires worker_count <= context_count; workers={}, contexts={}", getName(),
                  worker_count_, context_count_);
        return false;
    }
    if (config["queue_capacity"]) {
        queue_capacity_ = std::max<size_t>(1, config["queue_capacity"].as<size_t>());
    }
    queue_capacity_ = std::max(queue_capacity_, worker_count_ * 2);
    if (config["max_detections"]) {
        max_detections_ = std::max<size_t>(1, config["max_detections"].as<size_t>());
    }
    if (confidence_threshold_ < 0.0f || confidence_threshold_ > 1.0f || nms_threshold_ < 0.0f
        || nms_threshold_ > 1.0f) {
        LOG_ERROR("RKNN instance {} thresholds must be in [0,1]; confidence={}, nms={}", getName(),
                  confidence_threshold_, nms_threshold_);
        return false;
    }

    std::ifstream labels(config["label_path"].as<std::string>());
    for (std::string line; std::getline(labels, line);) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            labels_.push_back(line);
        }
    }
    if (labels_.empty()) {
        LOG_ERROR("RKNN instance {} could not load labels from {}", getName(), config["label_path"].as<std::string>());
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
        LOG_ERROR("Failed to initialize RKNN execution pool for {}", getName());
        return false;
    }
    LOG_INFO("RKNN instance {} configured with {} contexts and {} workers", getName(), context_count_, worker_count_);
    return true;
}

bool RknnYoloInstance::load_model(const std::string& path, std::vector<rknn_context>& contexts) {
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
    const auto size = static_cast<size_t>(end);
    model_data_.resize(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(model_data_.data()), static_cast<std::streamsize>(size))) {
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
        LOG_ERROR("Failed to create {} RKNN contexts for {}", context_count_, path);
        return false;
    }
    LOG_INFO("RKNN model {} uses NPU core mask {} ({})", path, core_mask_name_, core_policy_);
    return true;
}

bool RknnYoloInstance::query_tensors(rknn_context context) {
    rknn_input_output_num io_count{};
    int                   status = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_count, sizeof(io_count));
    if (status != RKNN_SUCC || io_count.n_input != 1 || io_count.n_output == 0) {
        LOG_ERROR("RKNN model {} has unsupported IO contract: status={}, inputs={}, outputs={}", model_path_, status,
                  io_count.n_input, io_count.n_output);
        return false;
    }

    input_attrs_.resize(io_count.n_input);
    output_attrs_.resize(io_count.n_output);
    for (uint32_t index = 0; index < io_count.n_input; ++index) {
        input_attrs_[index]       = {};
        input_attrs_[index].index = index;
        status = rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attrs_[index], sizeof(rknn_tensor_attr));
        if (status != RKNN_SUCC) {
            LOG_ERROR("Failed to query RKNN input {}: {}", index, status);
            return false;
        }
        LOG_INFO("RKNN input {} name={} shape={} format={} type={}", index, input_attrs_[index].name,
                 tensor_shape(input_attrs_[index]), static_cast<int>(input_attrs_[index].fmt),
                 static_cast<int>(input_attrs_[index].type));
    }
    for (uint32_t index = 0; index < io_count.n_output; ++index) {
        output_attrs_[index]       = {};
        output_attrs_[index].index = index;
        status = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[index], sizeof(rknn_tensor_attr));
        if (status != RKNN_SUCC) {
            LOG_ERROR("Failed to query RKNN output {}: {}", index, status);
            return false;
        }
        LOG_INFO("RKNN output {} name={} shape={} format={} type={} quant={}", index, output_attrs_[index].name,
                 tensor_shape(output_attrs_[index]), static_cast<int>(output_attrs_[index].fmt),
                 static_cast<int>(output_attrs_[index].type), static_cast<int>(output_attrs_[index].qnt_type));
    }

    const auto& input = input_attrs_.front();
    const bool  nchw  = input.fmt == RKNN_TENSOR_NCHW;
    if (input.n_dims != 4 || (input.fmt != RKNN_TENSOR_NCHW && input.fmt != RKNN_TENSOR_NHWC)
        || input.dims[nchw ? 1 : 3] != 3) {
        LOG_ERROR("RKNN model {} has unsupported input tensor {}", model_path_, tensor_shape(input));
        return false;
    }
    split_output_pairs_.clear();
    if (model_type_ == "YOLO_DFL_SPLIT") {
        if (output_attrs_.size() < 2) {
            LOG_ERROR("RKNN model {} requires DFL box/class outputs; got {} tensors", model_path_,
                      output_attrs_.size());
            return false;
        }
        std::vector<bool> used_class_outputs(output_attrs_.size(), false);
        for (size_t box_index = 0; box_index < output_attrs_.size(); ++box_index) {
            const auto& box = output_attrs_[box_index];
            if (box.n_dims != 4 || tensor_channels(box) < 8 || tensor_channels(box) % 4 != 0) {
                continue;
            }
            const uint32_t grid_h = tensor_height(box);
            const uint32_t grid_w = tensor_width(box);
            for (size_t class_index = 0; class_index < output_attrs_.size(); ++class_index) {
                const auto& classes = output_attrs_[class_index];
                if (class_index == box_index || used_class_outputs[class_index] || classes.n_dims != 4
                    || tensor_channels(classes) != labels_.size() || tensor_height(classes) != grid_h
                    || tensor_width(classes) != grid_w) {
                    continue;
                }
                split_output_pairs_.emplace_back(box_index, class_index);
                used_class_outputs[class_index] = true;
                break;
            }
        }
        if (split_output_pairs_.empty()) {
            LOG_ERROR("RKNN model {} has no compatible DFL box/class output pairs", model_path_);
            return false;
        }
        return true;
    }

    if (output_attrs_.size() != 1) {
        LOG_ERROR("RKNN model {} exposes {} outputs. This adapter expects a decoded flat YOLO output; use "
                  "YOLO_DFL_SPLIT for platform split-head models",
                  model_path_, output_attrs_.size());
        return false;
    }
    const size_t feature_count = detection_feature_count(output_attrs_.front());
    if (feature_count == 0) {
        LOG_ERROR("RKNN model {} has unsupported decoded YOLO output {}", model_path_,
                  tensor_shape(output_attrs_.front()));
        return false;
    }
    const size_t class_count = feature_count - 5;
    if (labels_.size() != class_count) {
        LOG_ERROR("RKNN instance {} has {} labels but output requires {} classes", getName(), labels_.size(),
                  class_count);
        return false;
    }
    return true;
}

bool RknnYoloInstance::commit(Job& job) {
    Job queued = job;
    return pool_.commit(
        [this, queued](rknn_context context) mutable {
            const auto begin = std::chrono::steady_clock::now();
            if (!process(queued, context)) {
                fail_job(queued, "RKNN inference failed");
                return;
            }
            const double elapsed =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            {
                std::lock_guard<std::mutex> lock(perf_mutex_);
                perf_time_ms_ += elapsed;
                perf_count_ += 1.0f;
            }
            try {
                queued.promise->set_value(true);
            } catch (const std::future_error&) {
            }
        },
        [this, queued]() mutable { fail_job(queued, "instance stopped before inference"); });
}

void RknnYoloInstance::start() {
    pool_.start();
    LOG_INFO("RKNN instance {} started", getName());
}

void RknnYoloInstance::stop() {
    pool_.stop();
}

bool RknnYoloInstance::supports_interval_reuse() const {
    return true;
}

std::tuple<double, float> RknnYoloInstance::get_perf() {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    auto                        result = std::make_tuple(perf_time_ms_, perf_count_);
    perf_time_ms_                      = 0.0;
    perf_count_                        = 0.0f;
    return result;
}

bool RknnYoloInstance::process(Job& job, rknn_context context) {
    if (!job.data || !job.data->get_frame_meta() || job.data->get_frame_meta()->frame.empty()) {
        LOG_ERROR("RKNN instance {} received an empty frame", getName());
        return false;
    }
    const auto& attr = input_attrs_.front();
    if (attr.n_dims != 4) {
        LOG_ERROR("RKNN input must be rank 4; got {}", tensor_shape(attr));
        return false;
    }
    const bool nchw         = attr.fmt == RKNN_TENSOR_NCHW;
    const int  input_height = static_cast<int>(attr.dims[nchw ? 2 : 1]);
    const int  input_width  = static_cast<int>(attr.dims[nchw ? 3 : 2]);
    const int  channels     = static_cast<int>(attr.dims[nchw ? 1 : 3]);
    if (channels != 3 || input_width <= 0 || input_height <= 0) {
        LOG_ERROR("Unsupported RKNN input shape {}", tensor_shape(attr));
        return false;
    }

    const cv::Mat&     source = job.data->get_frame_meta()->frame;
    LetterboxTransform transform;
    transform.scale =
        std::min(input_width / static_cast<float>(source.cols), input_height / static_cast<float>(source.rows));
    const int resized_width  = std::max(1, static_cast<int>(std::round(source.cols * transform.scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(source.rows * transform.scale)));
    transform.pad_x          = (input_width - resized_width) * 0.5f;
    transform.pad_y          = (input_height - resized_height) * 0.5f;

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(resized_width, resized_height));
    cv::Mat letterboxed(input_height, input_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterboxed(
        cv::Rect(static_cast<int>(transform.pad_x), static_cast<int>(transform.pad_y), resized_width, resized_height)));
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    rknn_input input{};
    input.index        = 0;
    input.type         = RKNN_TENSOR_UINT8;
    input.fmt          = RKNN_TENSOR_NHWC;
    input.size         = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
    input.buf          = rgb.data;
    input.pass_through = 0;

    int status = rknn_inputs_set(context, 1, &input);
    if (status == RKNN_SUCC) {
        status = rknn_run(context, nullptr);
    }
    if (status != RKNN_SUCC) {
        LOG_ERROR("RKNN execution failed for {} with status {}", getName(), status);
        return false;
    }

    std::vector<rknn_output> outputs(output_attrs_.size());
    for (uint32_t index = 0; index < outputs.size(); ++index) {
        outputs[index].index      = index;
        outputs[index].want_float = 1;
    }
    status = rknn_outputs_get(context, static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr);
    if (status != RKNN_SUCC) {
        LOG_ERROR("rknn_outputs_get failed for {} with status {}", getName(), status);
        return false;
    }

    auto targets = std::make_shared<object_meta::FrameTargetList>();
    bool decoded = false;
    if (model_type_ == "YOLO_DFL_SPLIT") {
        decoded = decode_dfl_split_output(outputs, transform, source.cols, source.rows, targets);
    } else {
        const size_t output_elements = tensor_element_count(output_attrs_[0]);
        if (output_elements == 0 || (outputs[0].size != 0 && outputs[0].size < output_elements * sizeof(float))) {
            LOG_ERROR("RKNN output buffer for {} is invalid: elements={}, bytes={}", getName(), output_elements,
                      outputs[0].size);
            rknn_outputs_release(context, static_cast<uint32_t>(outputs.size()), outputs.data());
            return false;
        }
        decoded = decode_flat_output(static_cast<const float*>(outputs[0].buf), output_attrs_[0], transform,
                                     source.cols, source.rows, targets);
    }
    rknn_outputs_release(context, static_cast<uint32_t>(outputs.size()), outputs.data());
    if (!decoded) {
        return false;
    }
    job.data->set_frame_target_list(getName(), targets);
    return true;
}

bool RknnYoloInstance::decode_flat_output(const float* output, const rknn_tensor_attr& attr,
                                          const LetterboxTransform& transform, int source_width, int source_height,
                                          object_meta::FrameTargetList::ptr& targets) {
    if (!output || attr.n_dims < 2) {
        LOG_ERROR("RKNN output is not a flat detection tensor: {}", tensor_shape(attr));
        return false;
    }

    const size_t feature_count = detection_feature_count(attr);
    const bool   transposed    = feature_count != attr.dims[attr.n_dims - 1];
    const size_t element_count = tensor_element_count(attr);
    if (feature_count < 6 || feature_count > 512 || element_count == 0 || element_count % feature_count != 0) {
        LOG_ERROR("Unsupported decoded YOLO output shape {} (elements={})", tensor_shape(attr), element_count);
        return false;
    }
    const size_t box_count   = element_count / feature_count;
    const size_t class_count = feature_count - 5;
    if (class_count == 0) {
        return false;
    }
    auto value = [&](size_t row, size_t column) {
        return transposed ? output[column * box_count + row] : output[row * feature_count + column];
    };
    std::vector<Candidate> candidates;
    candidates.reserve(std::min<size_t>(box_count, 1024));
    for (size_t row = 0; row < box_count; ++row) {
        const float objectness = value(row, 4);
        if (!std::isfinite(objectness) || objectness < confidence_threshold_) {
            continue;
        }
        int   class_id    = 0;
        float class_score = value(row, 5);
        for (size_t class_index = 1; class_index < class_count; ++class_index) {
            const float score = value(row, class_index + 5);
            if (score > class_score) {
                class_score = score;
                class_id    = static_cast<int>(class_index);
            }
        }
        const float confidence = objectness * class_score;
        if (!std::isfinite(confidence) || confidence < confidence_threshold_) {
            continue;
        }
        const float center_x = value(row, 0);
        const float center_y = value(row, 1);
        const float width    = value(row, 2);
        const float height   = value(row, 3);
        Candidate   candidate{
            (center_x - width * 0.5f - transform.pad_x) / transform.scale,
            (center_y - height * 0.5f - transform.pad_y) / transform.scale,
            (center_x + width * 0.5f - transform.pad_x) / transform.scale,
            (center_y + height * 0.5f - transform.pad_y) / transform.scale,
            confidence,
            class_id,
        };
        candidate.left   = std::clamp(candidate.left, 0.0f, static_cast<float>(source_width - 1));
        candidate.top    = std::clamp(candidate.top, 0.0f, static_cast<float>(source_height - 1));
        candidate.right  = std::clamp(candidate.right, 0.0f, static_cast<float>(source_width - 1));
        candidate.bottom = std::clamp(candidate.bottom, 0.0f, static_cast<float>(source_height - 1));
        if (candidate.right > candidate.left && candidate.bottom > candidate.top) {
            candidates.push_back(candidate);
        }
    }

    return emit_candidates(candidates, targets);
}

bool RknnYoloInstance::decode_dfl_split_output(const std::vector<rknn_output>& outputs,
                                               const LetterboxTransform& transform, int source_width, int source_height,
                                               object_meta::FrameTargetList::ptr& targets) {
    const auto& input        = input_attrs_.front();
    const bool  input_nchw   = input.fmt == RKNN_TENSOR_NCHW;
    const int   input_height = static_cast<int>(input.dims[input_nchw ? 2 : 1]);
    const int   input_width  = static_cast<int>(input.dims[input_nchw ? 3 : 2]);
    auto        value        = [](const float* data, const rknn_tensor_attr& attr, size_t channel, size_t y, size_t x) {
        const size_t channels = tensor_channels(attr);
        const size_t height   = tensor_height(attr);
        const size_t width    = tensor_width(attr);
        if (attr.fmt == RKNN_TENSOR_NHWC) {
            return data[(y * width + x) * channels + channel];
        }
        return data[(channel * height + y) * width + x];
    };
    auto sigmoid = [](float input_value) {
        if (input_value >= 0.0f) {
            return 1.0f / (1.0f + std::exp(-input_value));
        }
        const float exponent = std::exp(input_value);
        return exponent / (1.0f + exponent);
    };

    std::vector<Candidate> candidates;
    for (const auto& pair : split_output_pairs_) {
        const auto box_index   = pair.first;
        const auto class_index = pair.second;
        if (box_index >= outputs.size() || class_index >= outputs.size()) {
            LOG_ERROR("RKNN split output indexes for {} are invalid", getName());
            return false;
        }
        const auto&  box_attr       = output_attrs_[box_index];
        const auto&  class_attr     = output_attrs_[class_index];
        const size_t box_elements   = tensor_element_count(box_attr);
        const size_t class_elements = tensor_element_count(class_attr);
        if (box_elements == 0 || class_elements == 0 || outputs[box_index].buf == nullptr
            || outputs[class_index].buf == nullptr
            || (outputs[box_index].size != 0 && outputs[box_index].size < box_elements * sizeof(float))
            || (outputs[class_index].size != 0 && outputs[class_index].size < class_elements * sizeof(float))) {
            LOG_ERROR("RKNN split output buffers for {} are invalid", getName());
            return false;
        }
        const auto*  box_data   = static_cast<const float*>(outputs[box_index].buf);
        const auto*  class_data = static_cast<const float*>(outputs[class_index].buf);
        const size_t grid_h     = tensor_height(box_attr);
        const size_t grid_w     = tensor_width(box_attr);
        const size_t reg_max    = tensor_channels(box_attr) / 4;
        if (grid_h == 0 || grid_w == 0 || reg_max < 2 || tensor_channels(class_attr) != labels_.size()) {
            LOG_ERROR("RKNN split output contract for {} is invalid", getName());
            return false;
        }
        const float stride_x = input_width / static_cast<float>(grid_w);
        const float stride_y = input_height / static_cast<float>(grid_h);
        for (size_t y = 0; y < grid_h; ++y) {
            for (size_t x = 0; x < grid_w; ++x) {
                int   class_id   = 0;
                float confidence = 0.0f;
                for (size_t class_offset = 0; class_offset < labels_.size(); ++class_offset) {
                    const float raw_score = value(class_data, class_attr, class_offset, y, x);
                    const float score     = class_scores_logits_ ? sigmoid(raw_score) : raw_score;
                    if (score > confidence) {
                        confidence = score;
                        class_id   = static_cast<int>(class_offset);
                    }
                }
                if (!std::isfinite(confidence) || confidence < confidence_threshold_) {
                    continue;
                }
                float distances[4] = {};
                for (size_t side = 0; side < 4; ++side) {
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (size_t bin = 0; bin < reg_max; ++bin) {
                        maximum = std::max(maximum, value(box_data, box_attr, side * reg_max + bin, y, x));
                    }
                    float denominator = 0.0f;
                    for (size_t bin = 0; bin < reg_max; ++bin) {
                        const float weight = std::exp(value(box_data, box_attr, side * reg_max + bin, y, x) - maximum);
                        denominator += weight;
                        distances[side] += weight * static_cast<float>(bin);
                    }
                    distances[side] =
                        denominator > 0.0f && std::isfinite(denominator) ? distances[side] / denominator : 0.0f;
                }
                const float center_x = (static_cast<float>(x) + 0.5f) * stride_x;
                const float center_y = (static_cast<float>(y) + 0.5f) * stride_y;
                Candidate   candidate{
                    (center_x - distances[0] * stride_x - transform.pad_x) / transform.scale,
                    (center_y - distances[1] * stride_y - transform.pad_y) / transform.scale,
                    (center_x + distances[2] * stride_x - transform.pad_x) / transform.scale,
                    (center_y + distances[3] * stride_y - transform.pad_y) / transform.scale,
                    confidence,
                    class_id,
                };
                candidate.left   = std::clamp(candidate.left, 0.0f, static_cast<float>(source_width - 1));
                candidate.top    = std::clamp(candidate.top, 0.0f, static_cast<float>(source_height - 1));
                candidate.right  = std::clamp(candidate.right, 0.0f, static_cast<float>(source_width - 1));
                candidate.bottom = std::clamp(candidate.bottom, 0.0f, static_cast<float>(source_height - 1));
                if (candidate.right > candidate.left && candidate.bottom > candidate.top) {
                    candidates.push_back(candidate);
                }
            }
        }
    }
    return emit_candidates(candidates, targets);
}

bool RknnYoloInstance::emit_candidates(std::vector<Candidate>&            candidates,
                                       object_meta::FrameTargetList::ptr& targets) const {
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.confidence > rhs.confidence;
    });
    std::vector<Candidate> kept;
    for (const auto& candidate : candidates) {
        if (kept.size() >= max_detections_) {
            break;
        }
        bool suppressed = false;
        for (const auto& previous : kept) {
            if (candidate.class_id == previous.class_id
                && intersection_over_union(candidate, previous) > nms_threshold_) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
        }
    }
    for (const auto& candidate : kept) {
        const std::string label = candidate.class_id < static_cast<int>(labels_.size())
                                      ? labels_[candidate.class_id]
                                      : std::to_string(candidate.class_id);
        targets->targets.emplace_back(std::make_shared<object_meta::FrameBoxTarget>(
            candidate.left, candidate.top, candidate.right, candidate.bottom, candidate.confidence, candidate.class_id,
            label));
    }
    return true;
}

void RknnYoloInstance::fail_job(Job& job, const std::string& reason) {
    LOG_ERROR("RKNN instance {}: {}", getName(), reason);
    try {
        job.promise->set_value(false);
    } catch (const std::future_error&) {
    }
}

}  // namespace infer

namespace {
Register<infer::Instance, infer::RknnYoloInstance, std::string> register_rknn_yolo("rknn_yolo");
}
