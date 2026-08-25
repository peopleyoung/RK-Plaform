#pragma once

#include <rknn_api.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "FrameTarget.h"
#include "Instance.h"
#include "RknnExecutionPool.h"

namespace infer {

class RknnYoloInstance : public Instance {
public:
    explicit RknnYoloInstance(const std::string& name);
    ~RknnYoloInstance() override;

    bool                      init(YAML::Node config) override;
    bool                      commit(Job& job) override;
    void                      start() override;
    void                      stop() override;
    bool                      supports_interval_reuse() const override;
    std::tuple<double, float> get_perf() override;

    struct Candidate {
        float left;
        float top;
        float right;
        float bottom;
        float confidence;
        int   class_id;
    };

private:
    struct LetterboxTransform {
        float scale = 1.0f;
        float pad_x = 0.0f;
        float pad_y = 0.0f;
    };

    bool load_model(const std::string& path, std::vector<rknn_context>& contexts);
    bool query_tensors(rknn_context context);
    bool process(Job& job, rknn_context context);
    bool decode_dfl_split_output(const std::vector<rknn_output>& outputs, const LetterboxTransform& transform,
                                 int source_width, int source_height, object_meta::FrameTargetList::ptr& targets);
    bool emit_candidates(std::vector<Candidate>& candidates, object_meta::FrameTargetList::ptr& targets) const;
    void fail_job(Job& job, const std::string& reason);

    std::vector<uint8_t>                   model_data_;
    std::vector<rknn_tensor_attr>          input_attrs_;
    std::vector<rknn_tensor_attr>          output_attrs_;
    std::vector<std::pair<size_t, size_t>> split_output_pairs_;
    std::vector<std::string>               labels_;
    std::string                            model_path_;
    std::string                            model_type_;
    float                                  confidence_threshold_ = 0.4f;
    float                                  nms_threshold_        = 0.5f;
    bool                                   class_scores_logits_  = false;
    rknn_core_mask                         core_mask_            = RKNN_NPU_CORE_AUTO;
    std::string                            core_mask_name_       = "auto";
    std::string                            core_policy_          = "shared";
    size_t                                 context_count_        = 1;
    size_t                                 worker_count_         = 1;
    size_t                                 queue_capacity_       = 8;
    size_t                                 max_detections_       = 1024;

    RknnExecutionPool pool_;

    std::mutex perf_mutex_;
    double     perf_time_ms_ = 0.0;
    float      perf_count_   = 0.0f;
};

}  // namespace infer
