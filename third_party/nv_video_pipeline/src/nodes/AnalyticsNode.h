#pragma once

#include <json/value.h>
#include <opencv2/core/types.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ProcessNode.h"

namespace Node {

class AnalyticsNode : public GraphCore::Node {
public:
    explicit AnalyticsNode(const std::string& name);
    bool Init(YAML::Node config) override;
    Data::BaseData::ptr evaluate(Data::BaseData::ptr data);

private:
    struct AreaTrackState {
        bool     initialized{false};
        bool     stable_inside{false};
        bool     pending_inside{false};
        int      pending_frames{0};
        uint32_t last_seen_frame{0};
    };

    struct AreaRule {
        std::string                         id;
        std::string                         name;
        std::vector<cv::Point2f>            polygon;
        std::unordered_set<int>             class_ids;
        int                                 min_count{1};
        int                                 hold_frames{1};
        bool                                threshold_active{false};
        std::unordered_map<int, AreaTrackState> tracks;
    };

    struct LineTrackState {
        bool        initialized{false};
        cv::Point2f point;
        float       side{0.0F};
        uint32_t    last_seen_frame{0};
    };

    struct LineRule {
        std::string                         id;
        std::string                         name;
        cv::Point2f                         start;
        cv::Point2f                         end;
        std::string                         direction{"both"};
        std::unordered_set<int>             class_ids;
        uint64_t                            a_to_b_count{0};
        uint64_t                            b_to_a_count{0};
        std::unordered_set<int>             a_to_b_tracks;
        std::unordered_set<int>             b_to_a_tracks;
        std::unordered_map<int, LineTrackState> tracks;
    };

    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
    static bool         class_enabled(const std::unordered_set<int>& class_ids, int class_id);
    static float        line_side(const LineRule& line, const cv::Point2f& point);
    static bool         crosses_segment(const LineRule& line, const cv::Point2f& before,
                                        const cv::Point2f& after);
    static void         cleanup_stale_tracks(AreaRule& rule, uint32_t frame_index);
    static void         cleanup_stale_tracks(LineRule& rule, uint32_t frame_index);

    std::string           task_id_;
    std::string           primary_instance_;
    std::vector<AreaRule> areas_;
    std::vector<LineRule> lines_;
};

}  // namespace Node
