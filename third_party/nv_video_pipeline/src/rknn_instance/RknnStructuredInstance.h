#pragma once

#include <rknn_api.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "FrameInferenceResult.h"
#include "Instance.h"
#include "RknnExecutionPool.h"

namespace infer {

class RknnStructuredInstance final : public Instance {
public:
    explicit RknnStructuredInstance(const std::string& name);
    ~RknnStructuredInstance() override;

    bool                      init(YAML::Node config) override;
    bool                      commit(Job& job) override;
    void                      start() override;
    void                      stop() override;
    std::tuple<double, float> get_perf() override;

private:
    bool load_model(const std::string& path, std::vector<rknn_context>& contexts);
    bool query_tensors(rknn_context context);
    bool process(Job& job, rknn_context context);
    bool decode_deeplab(const float* output, const rknn_tensor_attr& attr, const object_meta::FrameMeta& frame,
                        object_meta::FrameInferenceResult::ptr& result);
    bool decode_ppocr_db(const float* output, const rknn_tensor_attr& attr, const object_meta::FrameMeta& frame,
                         object_meta::FrameInferenceResult::ptr& result);
    bool decode_ppocr_ctc(const float* output, const rknn_tensor_attr& attr,
                          object_meta::FrameInferenceResult::ptr& result);
    void fail_job(Job& job, const std::string& reason);

    static size_t      tensor_element_count(const rknn_tensor_attr& attr);
    static uint32_t    tensor_channels(const rknn_tensor_attr& attr);
    static uint32_t    tensor_height(const rknn_tensor_attr& attr);
    static uint32_t    tensor_width(const rknn_tensor_attr& attr);
    static std::string tensor_shape(const rknn_tensor_attr& attr);

    std::vector<uint8_t>          model_data_;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<std::string>      labels_;
    std::string                   model_path_;
    std::string                   model_type_;
    int                           blank_index_       = 0;
    bool                          ctc_scores_logits_ = false;
    rknn_core_mask                core_mask_         = RKNN_NPU_CORE_AUTO;
    std::string                   core_mask_name_    = "auto";
    std::string                   core_policy_       = "shared";
    float                         binary_threshold_  = 0.3f;
    float                         box_threshold_     = 0.6f;
    float                         unclip_ratio_      = 1.5f;
    int                           min_size_          = 3;
    size_t                        max_candidates_    = 1000;
    size_t                        max_regions_       = 100;
    size_t                        context_count_     = 1;
    size_t                        worker_count_      = 1;
    size_t                        queue_capacity_    = 8;

    RknnExecutionPool pool_;

    std::mutex perf_mutex_;
    double     perf_time_ms_ = 0.0;
    float      perf_count_   = 0.0f;
};

}  // namespace infer
