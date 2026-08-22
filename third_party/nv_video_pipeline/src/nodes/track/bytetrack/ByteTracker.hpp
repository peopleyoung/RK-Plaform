#pragma once
#include <opencv2/core/types.hpp>
#include <vector>

#include "FrameTarget.h"
#include "Strack.hpp"

namespace ByteTrack {

struct Object {
    cv::Rect_<float>              rect;
    float                         score;
    object_meta::FrameTarget::ptr frame_target;
};

class BYTETracker {
public:
    BYTETracker(int frame_rate = 30, int track_buffer = 30);
    ~BYTETracker();

    vector<STrack> update(const std::vector<Object> &objects);
    Scalar         get_color(int idx);
    void           reset_parameters(int frame_rate, int track_buffer);

private:
    vector<STrack *> joint_stracks(vector<STrack *> &tlista, vector<STrack> &tlistb);
    vector<STrack>   joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

    vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
    void           remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa,
                                            vector<STrack> &stracksb);

    void linear_assignment(vector<vector<float>> &cost_matrix, int cost_matrix_size, int cost_matrix_size_size,
                           float thresh, vector<vector<int>> &matches, vector<int> &unmatched_a,
                           vector<int> &unmatched_b);
    vector<vector<float>> iou_distance(vector<STrack *> &atracks, vector<STrack> &btracks, int &dist_size,
                                       int &dist_size_size);
    vector<vector<float>> iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
    vector<vector<float>> ious(vector<vector<float>> &atlbrs, vector<vector<float>> &btlbrs);

    double lapjv(const vector<vector<float>> &cost, vector<int> &rowsol, vector<int> &colsol, bool extend_cost = false,
                 float cost_limit = LONG_MAX, bool return_cost = true);

private:
    float track_thresh;
    float high_thresh;
    float match_thresh;
    int   frame_id;
    int   max_time_lost;

    vector<STrack> tracked_stracks;
    vector<STrack> lost_stracks;
    vector<STrack> removed_stracks;
    KalmanFilter   kalman_filter;
};
}  // namespace ByteTrack
