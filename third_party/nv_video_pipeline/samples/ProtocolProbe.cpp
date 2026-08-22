#include <json/json.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>

#include "AnhuanMessage.h"
#include "AnalyticsNode.h"
#include "BaseData.h"
#include "FrameBoxTarget.h"
#include "FrameMeta.h"
#include "FrameInferenceResult.h"
#include "SeiPacket.h"
#include "track/bytetrack/ByteTracker.hpp"

int main() {
    ByteTrack::BYTETracker tracker(25, 30);
    auto tracked_target = std::make_shared<object_meta::FrameBoxTarget>(1, 2, 9, 12, 0.9F, 3, "part");
    ByteTrack::Object tracked_object{
        cv::Rect_<float>(1.0F, 2.0F, 8.0F, 10.0F), 0.9F, tracked_target};
    const auto first_tracks = tracker.update({tracked_object});
    const auto second_tracks = tracker.update({tracked_object});
    if (first_tracks.size() != 1 || second_tracks.size() != 1 || first_tracks[0].track_id <= 0 ||
        first_tracks[0].track_id != second_tracks[0].track_id) {
        std::cerr << "ByteTrack identity mismatch" << std::endl;
        return 1;
    }

    auto data = std::make_shared<Data::BaseData>();
    data->set_frame_meta(std::make_shared<object_meta::FrameMeta>(cv::Mat::zeros(16, 16, CV_8UC3), 7));
    auto targets = std::make_shared<object_meta::FrameTargetList>();
    auto box = std::make_shared<object_meta::FrameBoxTarget>(1, 2, 9, 12, 0.9F, 3, "part");
    box->track_id = 42;
    targets->targets.push_back(box);
    data->set_frame_target_list("detector", targets);
    std::vector<object_meta::OcrRegion> regions;
    regions.push_back(object_meta::OcrRegion{{cv::Point2f(1, 1), cv::Point2f(8, 1),
                                               cv::Point2f(8, 6), cv::Point2f(1, 6)},
                                              0.88F, "A12"});
    data->set_frame_inference_result(
        "ocr", std::make_shared<object_meta::FrameOcrResult>("ocr_detection", regions));
    Json::Value analytics;
    analytics["areas"][0]["id"] = "zone-a";
    analytics["areas"][0]["count"] = 1;
    analytics["lines"][0]["id"] = "gate-a";
    analytics["lines"][0]["a_to_b_count"] = Json::UInt64(2);
    analytics["events"][0]["type"] = "line_cross";
    data->set_analytics_result(analytics);
    Json::Value media;
    media["event_root"] = "/data/output/events/task-a";
    data->set_media_result(media);

    const std::string message = Node::build_anhuan_message(data, "task-a", 9, "detector");
    Json::Value parsed;
    Json::CharReaderBuilder reader;
    std::string error;
    std::istringstream input(message);
    if (!Json::parseFromStream(reader, input, &parsed, &error) ||
        parsed["schema_version"].asInt() != 2 || parsed["task_id"].asString() != "task-a" ||
        parsed["revision"].asUInt64() != 9 || parsed["index"].asUInt() != 7 ||
        parsed["frame_index"].asUInt() != 7 || parsed["instance"].asString() != "detector" ||
        parsed["detector"]["datas"]["object"][0]["track_id"].asInt() != 42 ||
        parsed["detections"][0]["class_id"].asInt() != 3 ||
        parsed["detection_results"]["detector"][0]["class_id"].asInt() != 3 ||
        parsed["structured_results"]["ocr"]["type"].asString() != "ocr_detection" ||
        parsed["analytics"]["areas"][0]["count"].asInt() != 1 ||
        parsed["analytics"]["events"][0]["type"].asString() != "line_cross" ||
        parsed["media"]["event_root"].asString() != "/data/output/events/task-a") {
        std::cerr << "anhuan_v1 message mismatch: " << error << std::endl;
        return 2;
    }
    if (Node::default_anhuan_key("rtsp://zlm/live/camera-01?token=x") != "live_camera-01") {
        std::cerr << "Kafka key mismatch" << std::endl;
        return 3;
    }
    const auto h264 = media::make_user_data_sei(message, object_meta::VideoCodec::H264);
    const auto h265 = media::make_user_data_sei(message, object_meta::VideoCodec::H265);
    const std::vector<uint8_t> h264_prefix = {0, 0, 0, 1, 6, 5};
    const std::vector<uint8_t> h265_prefix = {0, 0, 0, 1, 0x4e, 1, 5};
    if (!h264 || !h265
        || !std::equal(h264_prefix.begin(), h264_prefix.end(), h264->begin())
        || !std::equal(h265_prefix.begin(), h265_prefix.end(), h265->begin())
        || h264->back() != 0x80 || h265->back() != 0x80) {
        std::cerr << "SEI packet mismatch" << std::endl;
        return 4;
    }

    YAML::Node analytics_config = YAML::Load(R"(
primary_instance: detector
task_id: task-a
areas:
  - id: zone-a
    name: zone-a
    minCount: 1
    holdFrames: 1
    polygon:
      - {x: 0.4, y: 0.2}
      - {x: 0.9, y: 0.2}
      - {x: 0.9, y: 0.9}
      - {x: 0.4, y: 0.9}
lines:
  - id: gate-a
    name: gate-a
    direction: both
    start: {x: 0.5, y: 0.1}
    end: {x: 0.5, y: 0.9}
)");
    Node::AnalyticsNode analytics_node("analytics_probe");
    if (!analytics_node.Init(analytics_config)) {
        std::cerr << "Analytics initialization mismatch" << std::endl;
        return 5;
    }
    const auto evaluate = [&analytics_node](uint32_t frame_index, float center_x) {
        auto frame_data = std::make_shared<Data::BaseData>();
        frame_data->set_frame_meta(
            std::make_shared<object_meta::FrameMeta>(cv::Mat::zeros(100, 100, CV_8UC3), frame_index));
        auto frame_targets = std::make_shared<object_meta::FrameTargetList>();
        auto frame_box = std::make_shared<object_meta::FrameBoxTarget>(
            center_x - 5.0F, 40.0F, center_x + 5.0F, 60.0F, 0.9F, 0, "part");
        frame_box->track_id = 7;
        frame_targets->targets.push_back(frame_box);
        frame_data->set_frame_target_list("detector", frame_targets);
        return analytics_node.evaluate(frame_data)->get_analytics_result();
    };
    evaluate(1, 20.0F);
    const Json::Value business = evaluate(2, 60.0F);
    if (business["areas"][0]["count"].asInt() != 1 ||
        business["lines"][0]["b_to_a_count"].asUInt64() != 1 ||
        business["events"].size() < 2) {
        std::cerr << "Analytics state machine mismatch" << std::endl;
        return 6;
    }
    std::cout << message << std::endl;
    return 0;
}
