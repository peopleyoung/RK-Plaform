#include "EventOutputNode.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <json/json.h>
#include <opencv2/imgcodecs.hpp>
#include <system_error>

#include "BaseData.h"
#include "CLog.h"
#include "Register.h"
#include "StatusCode.h"

namespace Node {

EventOutputNode::EventOutputNode(const std::string& name) : GraphCore::Node(name) {
    m_type = GraphCore::MID_NODE;
}

EventOutputNode::~EventOutputNode() {
    stop_recording();
}

bool EventOutputNode::Init(YAML::Node config) {
    CHECK(GraphCore::Node::Init(config), "EventOutputNode base initialization failed");
    CHECK(config["output"], "EventOutputNode requires output");
    task_id_ = config["task_id"] ? config["task_id"].as<std::string>() : "";
    output_root_ = config["output"].as<std::string>();
    snapshot_enabled_ = !config["snapshot"] || config["snapshot"].as<bool>();
    record_enabled_   = config["record"] && config["record"].as<bool>();
    pre_seconds_      = config["pre_seconds"] ? config["pre_seconds"].as<int>() : 3;
    post_seconds_     = config["post_seconds"] ? config["post_seconds"].as<int>() : 5;
    retention_days_   = config["retention_days"] ? config["retention_days"].as<int>() : 30;
    CHECK(pre_seconds_ >= 0 && pre_seconds_ <= 60,
          "EventOutputNode pre_seconds is out of range");
    CHECK(post_seconds_ >= 0 && post_seconds_ <= 300,
          "EventOutputNode post_seconds is out of range");
    CHECK(retention_days_ >= 1 && retention_days_ <= 3650,
          "EventOutputNode retention_days is out of range");
    std::filesystem::create_directories(output_root_ / "snapshots");
    std::filesystem::create_directories(output_root_ / "clips");
    cleanup_retention();
    last_retention_cleanup_ = std::chrono::steady_clock::now();
    event_log_.open(output_root_ / "events.jsonl", std::ios::out | std::ios::app);
    CHECK(event_log_, "EventOutputNode could not open event log");
    return true;
}

void EventOutputNode::cleanup_retention() {
    const auto cutoff = std::filesystem::file_time_type::clock::now() -
                        std::chrono::hours(24 * retention_days_);
    for (const auto& directory : {output_root_ / "snapshots", output_root_ / "clips"}) {
        std::error_code iterator_error;
        std::filesystem::directory_iterator iterator(directory, iterator_error);
        const std::filesystem::directory_iterator end;
        while (!iterator_error && iterator != end) {
            const auto path = iterator->path();
            std::error_code status_error;
            const bool removable = iterator->is_regular_file(status_error) && !status_error &&
                                   iterator->last_write_time(status_error) < cutoff && !status_error;
            if (removable) {
                std::filesystem::remove(path, status_error);
                if (status_error) {
                    LOG_WARN("EventOutputNode {} could not remove expired file {}: {}", getName(),
                             path.string(), status_error.message());
                }
            }
            iterator.increment(iterator_error);
        }
    }
}

std::string EventOutputNode::safe_filename(const std::string& value) {
    std::string result;
    result.reserve(std::min<size_t>(value.size(), 96));
    for (const unsigned char character : value) {
        if (std::isalnum(character) || character == '-' || character == '_') {
            result.push_back(static_cast<char>(character));
        } else {
            result.push_back('-');
        }
        if (result.size() >= 96) {
            break;
        }
    }
    return result.empty() ? "event" : result;
}

void EventOutputNode::prune_buffer() {
    if (buffer_.empty()) {
        return;
    }
    const int64_t newest = buffer_.back().packet->pts_us;
    const int64_t cutoff = newest - static_cast<int64_t>(pre_seconds_) * 1000000;
    while (buffer_.size() > 1 && buffer_.front().packet->pts_us < cutoff) {
        buffer_.pop_front();
    }
    int fps = buffer_.back().meta && buffer_.back().meta->fps > 0 ? buffer_.back().meta->fps : 30;
    const size_t maximum = static_cast<size_t>(std::max(1, fps * std::max(1, pre_seconds_ + 1)));
    while (buffer_.size() > maximum) {
        buffer_.pop_front();
    }
}

bool EventOutputNode::save_snapshot(const Data::BaseData::ptr& data,
                                    const std::string& event_id,
                                    std::filesystem::path& result) {
    const auto frame = data->get_frame_meta();
    if (!frame || frame->frame.empty()) {
        return false;
    }
    result = output_root_ / "snapshots" / (safe_filename(event_id) + ".jpg");
    return cv::imwrite(result.string(), frame->frame,
                       {cv::IMWRITE_JPEG_QUALITY, 90});
}

bool EventOutputNode::start_recording(const std::string& event_id) {
    stop_recording();
    if (buffer_.empty()) {
        return false;
    }
    auto first = std::find_if(buffer_.begin(), buffer_.end(), [](const BufferedPacket& item) {
        return item.packet && item.packet->key_frame;
    });
    if (first == buffer_.end()) {
        LOG_WARN("EventOutputNode {} is waiting for a key frame before recording", getName());
        return false;
    }
    record_path_ = output_root_ / "clips" / (safe_filename(event_id) + ".ts");
    if (avformat_alloc_output_context2(&record_format_, nullptr, "mpegts",
                                      record_path_.c_str()) < 0 || !record_format_) {
        return false;
    }
    record_stream_ = avformat_new_stream(record_format_, nullptr);
    if (!record_stream_) {
        stop_recording();
        return false;
    }
    const auto& meta = first->meta;
    const auto& packet = first->packet;
    record_stream_->time_base            = AVRational{1, 1000000};
    record_stream_->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    record_stream_->codecpar->codec_id = packet->codec == object_meta::VideoCodec::H265
                                              ? AV_CODEC_ID_HEVC
                                              : AV_CODEC_ID_H264;
    record_stream_->codecpar->width  = meta ? meta->width : 0;
    record_stream_->codecpar->height = meta ? meta->height : 0;
    record_stream_->codecpar->bit_rate = meta ? meta->bitrate : 0;
    if (!packet->codec_extradata.empty()) {
        record_stream_->codecpar->extradata = static_cast<uint8_t*>(
            av_mallocz(packet->codec_extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE));
        if (!record_stream_->codecpar->extradata) {
            stop_recording();
            return false;
        }
        std::memcpy(record_stream_->codecpar->extradata, packet->codec_extradata.data(),
                    packet->codec_extradata.size());
        record_stream_->codecpar->extradata_size =
            static_cast<int>(packet->codec_extradata.size());
    }
    if (!(record_format_->oformat->flags & AVFMT_NOFILE) &&
        avio_open(&record_format_->pb, record_path_.c_str(), AVIO_FLAG_WRITE) < 0) {
        stop_recording();
        return false;
    }
    if (avformat_write_header(record_format_, nullptr) < 0) {
        stop_recording();
        return false;
    }
    record_header_written_ = true;
    record_base_pts_us_ = first->packet->pts_us;
    last_written_pts_us_ = INT64_MIN;
    for (auto item = first; item != buffer_.end(); ++item) {
        if (!write_packet(*item)) {
            stop_recording();
            return false;
        }
    }
    LOG_INFO("EventOutputNode {} started zero-reencode clip {}", getName(),
             record_path_.string());
    return true;
}

bool EventOutputNode::write_packet(const BufferedPacket& item) {
    if (!record_format_ || !record_stream_ || !item.packet || item.packet->bytes.empty()) {
        return false;
    }
    if (item.packet->pts_us == last_written_pts_us_) {
        return true;
    }
    AVPacket* output = av_packet_alloc();
    if (!output || av_new_packet(output, static_cast<int>(item.packet->bytes.size())) < 0) {
        av_packet_free(&output);
        return false;
    }
    std::copy(item.packet->bytes.begin(), item.packet->bytes.end(), output->data);
    output->stream_index = record_stream_->index;
    output->pts          = std::max<int64_t>(0, item.packet->pts_us - record_base_pts_us_);
    output->dts          = std::max<int64_t>(0, item.packet->dts_us - record_base_pts_us_);
    output->duration     = item.packet->duration_us;
    if (item.packet->key_frame) {
        output->flags |= AV_PKT_FLAG_KEY;
    }
    const int status = av_interleaved_write_frame(record_format_, output);
    av_packet_free(&output);
    if (status < 0) {
        LOG_WARN("EventOutputNode {} clip write failed: {}", getName(), status);
        return false;
    }
    last_written_pts_us_ = item.packet->pts_us;
    return true;
}

void EventOutputNode::stop_recording() {
    if (!record_format_) {
        return;
    }
    if (record_header_written_) {
        av_write_trailer(record_format_);
    }
    if (!(record_format_->oformat->flags & AVFMT_NOFILE) && record_format_->pb) {
        avio_closep(&record_format_->pb);
    }
    avformat_free_context(record_format_);
    record_format_ = nullptr;
    record_stream_ = nullptr;
    record_header_written_ = false;
    recording_until_us_ = 0;
    record_base_pts_us_ = 0;
    last_written_pts_us_ = INT64_MIN;
    LOG_INFO("EventOutputNode {} completed clip {}", getName(), record_path_.string());
}

Data::BaseData::ptr EventOutputNode::handle_data(Data::BaseData::ptr data) {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_retention_cleanup_ >= std::chrono::hours(1)) {
        cleanup_retention();
        last_retention_cleanup_ = now;
    }

    const auto packet_meta = data->get_packet_meta();
    if (packet_meta && packet_meta->encoded_packet &&
        !packet_meta->encoded_packet->bytes.empty()) {
        buffer_.push_back(BufferedPacket{packet_meta, packet_meta->encoded_packet});
        prune_buffer();
    }

    Json::Value analytics = data->get_analytics_result();
    Json::Value events = analytics.isObject() ? analytics["events"]
                                              : Json::Value(Json::arrayValue);
    Json::Value media = data->get_media_result();
    if (!media.isObject()) {
        media = Json::Value(Json::objectValue);
    }
    media["event_root"] = output_root_.string();
    Json::Value emitted(Json::arrayValue);
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    if (events.isArray()) {
        for (const auto& event : events) {
            Json::Value output_event = event;
            const std::string event_id = event["id"].asString();
            if (snapshot_enabled_) {
                std::filesystem::path snapshot;
                if (save_snapshot(data, event_id, snapshot)) {
                    output_event["snapshot"] = snapshot.string();
                }
            }
            if (record_enabled_ && packet_meta && packet_meta->encoded_packet) {
                if (!record_format_) {
                    start_recording(event_id);
                }
                if (record_format_) {
                    recording_until_us_ = std::max(
                        recording_until_us_,
                        packet_meta->encoded_packet->pts_us +
                            static_cast<int64_t>(post_seconds_) * 1000000);
                    output_event["clip"] = record_path_.string();
                }
            }
            event_log_ << Json::writeString(writer, output_event) << '\n';
            emitted.append(std::move(output_event));
        }
        event_log_.flush();
    }
    if (!emitted.empty()) {
        media["events"] = std::move(emitted);
    }
    if (record_format_ && packet_meta && packet_meta->encoded_packet) {
        const BufferedPacket current{packet_meta, packet_meta->encoded_packet};
        if (!write_packet(current) || packet_meta->encoded_packet->pts_us >= recording_until_us_) {
            stop_recording();
        }
    }
    data->set_media_result(std::move(media));
    return data;
}

}  // namespace Node

namespace {
Register<GraphCore::Node, Node::EventOutputNode, std::string>
    register_event_output("EventOutputNode");
}
