#include "ZlmSeiOutputNode.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "AnhuanMessage.h"
#include "CLog.h"
#include "FrameInferenceResult.h"
#include "MediaUrl.h"
#include "Register.h"
#include "SeiPacket.h"
#include "StatusCode.h"

namespace Node {

ZlmSeiOutputNode::ZlmSeiOutputNode(const std::string& name) : GraphCore::Node(name) {
    m_type = GraphCore::DES_NODE;
}

ZlmSeiOutputNode::~ZlmSeiOutputNode() {
    close();
}

bool ZlmSeiOutputNode::Init(YAML::Node config) {
    CHECK(GraphCore::Node::Init(config), "ZlmSeiOutputNode base initialization failed");
    CHECK(config["output"], "ZlmSeiOutputNode requires output");
    output_uri_ = config["output"].as<std::string>();
    task_id_       = config["task_id"] ? config["task_id"].as<std::string>() : "";
    instance_name_ = config["instance"] ? config["instance"].as<std::string>() : "";
    revision_      = config["revision"] ? config["revision"].as<uint64_t>() : 0;
    CHECK(output_uri_.rfind("rtsp://", 0) == 0, "ZlmSeiOutputNode output must use rtsp://");
    if (config["reconnect_ms"]) {
        reconnect_ms_ = config["reconnect_ms"].as<int>();
    }
    CHECK(reconnect_ms_ >= 1000 && reconnect_ms_ <= 4000,
          "ZLM reconnect_ms must be between 1000 and 4000");
    return true;
}

bool ZlmSeiOutputNode::open(const object_meta::PacketMeta::ptr& packet) {
    close();
    codec_ = packet->encoded_packet->codec;
    if (avformat_alloc_output_context2(&format_, nullptr, "rtsp", output_uri_.c_str()) < 0 || !format_) {
        LOG_WARN("ZlmSeiOutputNode {} could not allocate RTSP output {}", getName(),
                 media::redact_media_url(output_uri_));
        return false;
    }
    stream_ = avformat_new_stream(format_, nullptr);
    if (!stream_) {
        close();
        return false;
    }
    stream_->time_base                = AVRational{1, 1000000};
    stream_->codecpar->codec_type     = AVMEDIA_TYPE_VIDEO;
    stream_->codecpar->codec_id       = codec_ == object_meta::VideoCodec::H265 ? AV_CODEC_ID_HEVC : AV_CODEC_ID_H264;
    stream_->codecpar->width          = packet->width;
    stream_->codecpar->height         = packet->height;
    const auto& extradata = packet->encoded_packet->codec_extradata;
    if (!extradata.empty()) {
        stream_->codecpar->extradata = static_cast<uint8_t*>(
            av_mallocz(extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE));
        if (!stream_->codecpar->extradata) {
            close();
            return false;
        }
        std::memcpy(stream_->codecpar->extradata, extradata.data(), extradata.size());
        stream_->codecpar->extradata_size = static_cast<int>(extradata.size());
    }
    AVDictionary* options             = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "muxdelay", "0", 0);
    const int result = avformat_write_header(format_, &options);
    av_dict_free(&options);
    if (result < 0) {
        LOG_WARN("ZlmSeiOutputNode {} could not connect to {}: {}", getName(),
                 media::redact_media_url(output_uri_), result);
        close();
        return false;
    }
    header_written_ = true;
    LOG_INFO("ZlmSeiOutputNode {} publishing unchanged encoded video to {}", getName(),
             media::redact_media_url(output_uri_));
    return true;
}

void ZlmSeiOutputNode::close() {
    if (!format_) {
        return;
    }
    if (header_written_) {
        av_write_trailer(format_);
    }
    if (!(format_->oformat->flags & AVFMT_NOFILE) && format_->pb) {
        avio_closep(&format_->pb);
    }
    avformat_free_context(format_);
    format_ = nullptr;
    stream_ = nullptr;
    header_written_ = false;
}

void ZlmSeiOutputNode::schedule_retry(std::chrono::steady_clock::time_point now) {
    if (publish_failure_count_ < std::numeric_limits<uint64_t>::max()) {
        ++publish_failure_count_;
    }
    const uint32_t exponent = std::min<uint32_t>(consecutive_failures_, 2);
    const int delay_ms = std::min(4000, reconnect_ms_ * (1 << exponent));
    if (consecutive_failures_ < 2) {
        ++consecutive_failures_;
    }
    retry_after_ = now + std::chrono::milliseconds(delay_ms);
}

bool ZlmSeiOutputNode::metadata_within_limits(const Data::BaseData::ptr& data) const {
    if (instance_name_.empty() || !data->has_frame_inference_result(instance_name_)) {
        return true;
    }
    const auto result = data->get_frame_inference_result(instance_name_);
    const auto segmentation =
        std::dynamic_pointer_cast<object_meta::FrameSegmentationResult>(result);
    return !segmentation
           || media::sei_metadata_within_limits(segmentation->source_width(),
                                                segmentation->source_height(),
                                                segmentation->run_count());
}

void ZlmSeiOutputNode::warn_sei_skipped() {
    if (sei_skipped_count_ < std::numeric_limits<uint64_t>::max()) {
        ++sei_skipped_count_;
    }
    const auto now = std::chrono::steady_clock::now();
    if (last_sei_warning_.time_since_epoch().count() == 0
        || now - last_sei_warning_ >= std::chrono::seconds(60)) {
        LOG_WARN("ZlmSeiOutputNode {} skipped oversized SEI metadata; skipped={}",
                 getName(), sei_skipped_count_);
        last_sei_warning_ = now;
    }
}

Data::BaseData::ptr ZlmSeiOutputNode::handle_data(Data::BaseData::ptr data) {
    const auto packet = data->get_packet_meta();
    if (!packet || !packet->encoded_packet || packet->encoded_packet->bytes.empty()) {
        LOG_WARN("ZlmSeiOutputNode {} received a frame without an encoded packet", getName());
        return data;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!format_) {
        if (now < retry_after_) {
            return data;
        }
        if (!packet->encoded_packet->key_frame) {
            return data;
        }
        if (!open(packet)) {
            schedule_retry(now);
            return data;
        }
    }
    const auto& source = packet->encoded_packet->bytes;
    std::vector<uint8_t> access_unit;
    std::optional<std::vector<uint8_t>> sei;
    if (metadata_within_limits(data)) {
        sei = media::make_user_data_sei(
            build_anhuan_message(data, task_id_, revision_, instance_name_),
            packet->encoded_packet->codec);
    }
    if (!sei) {
        warn_sei_skipped();
    }
    access_unit.reserve((sei ? sei->size() : 0) + source.size());
    if (sei) {
        access_unit.insert(access_unit.end(), sei->begin(), sei->end());
    }
    access_unit.insert(access_unit.end(), source.begin(), source.end());

    AVPacket* output = av_packet_alloc();
    if (!output || av_new_packet(output, static_cast<int>(access_unit.size())) < 0) {
        av_packet_free(&output);
        return data;
    }
    std::copy(access_unit.begin(), access_unit.end(), output->data);
    output->stream_index = stream_->index;
    output->pts          = packet->encoded_packet->pts_us;
    output->dts          = packet->encoded_packet->dts_us;
    output->duration     = packet->encoded_packet->duration_us;
    if (packet->encoded_packet->key_frame) {
        output->flags |= AV_PKT_FLAG_KEY;
    }
    const int result = av_interleaved_write_frame(format_, output);
    av_packet_free(&output);
    if (result < 0) {
        LOG_WARN("ZlmSeiOutputNode {} RTSP write to {} failed: {}", getName(),
                 media::redact_media_url(output_uri_), result);
        close();
        schedule_retry(now);
    } else {
        consecutive_failures_ = 0;
        retry_after_ = {};
    }
    return data;
}

}  // namespace Node

namespace {
Register<GraphCore::Node, Node::ZlmSeiOutputNode, std::string> register_zlm_sei_output("ZlmSeiOutputNode");
}
