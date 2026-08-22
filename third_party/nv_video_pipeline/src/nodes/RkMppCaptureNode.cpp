#include "RkMppCaptureNode.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

extern "C" {
#include <libavutil/avutil.h>
}
#include <opencv2/imgproc.hpp>

#include "BaseData.h"
#include "CLog.h"
#include "FrameMeta.h"
#include "Register.h"
#include "StatusCode.h"

namespace Node {
namespace {
int interrupt_io(void* opaque) {
    return static_cast<std::atomic_bool*>(opaque)->load() ? 0 : 1;
}

int64_t to_us(int64_t value, AVRational time_base) {
    return value == AV_NOPTS_VALUE ? 0 : av_rescale_q(value, time_base, AVRational{1, 1000000});
}
}

RkMppCaptureNode::RkMppCaptureNode(const std::string& name) : GraphCore::Node(name) {
    m_type = GraphCore::SRC_NODE;
    set_after_start_cb([this] { return after_start(); });
    set_exit_cb([this] { return on_exit(); });
}

RkMppCaptureNode::~RkMppCaptureNode() {
    on_exit();
}

bool RkMppCaptureNode::Init(YAML::Node config) {
    CHECK(GraphCore::Node::Init(config), "RkMppCaptureNode base initialization failed");
    CHECK(config["input"], "RkMppCaptureNode requires input");
    input_ = config["input"].as<std::string>();
    CHECK(input_.rfind("rtsp://", 0) == 0, "RkMppCaptureNode requires an rtsp:// input");
    if (config["reconnect_ms"]) {
        reconnect_ms_ = config["reconnect_ms"].as<int>();
    }
    if (config["open_timeout_ms"]) {
        open_timeout_ms_ = config["open_timeout_ms"].as<int>();
    }
    CHECK(reconnect_ms_ >= 100 && reconnect_ms_ <= 60000, "RkMppCaptureNode reconnect_ms is out of range");
    CHECK(open_timeout_ms_ >= 100 && open_timeout_ms_ <= 60000, "RkMppCaptureNode open timeout is out of range");
    return true;
}

bool RkMppCaptureNode::init_decoder(AVCodecID codec) {
    close_decoder();
    const MppCodingType coding = codec == AV_CODEC_ID_HEVC ? MPP_VIDEO_CodingHEVC : MPP_VIDEO_CodingAVC;
    codec_                     = codec == AV_CODEC_ID_HEVC ? object_meta::VideoCodec::H265
                                                           : object_meta::VideoCodec::H264;
    if (mpp_create(&decoder_, &decoder_api_) != MPP_OK) {
        return false;
    }
    RK_U32 split = 1;
    if (decoder_api_->control(decoder_, MPP_DEC_SET_PARSER_SPLIT_MODE, &split) != MPP_OK) {
        close_decoder();
        return false;
    }
    if (mpp_init(decoder_, MPP_CTX_DEC, coding) != MPP_OK) {
        close_decoder();
        return false;
    }
    MppFrameFormat output_format = MPP_FMT_YUV420SP;
    RK_S64         timeout       = 0;
    if (decoder_api_->control(decoder_, MPP_DEC_SET_OUTPUT_FORMAT, &output_format) != MPP_OK ||
        decoder_api_->control(decoder_, MPP_SET_OUTPUT_TIMEOUT, &timeout) != MPP_OK) {
        close_decoder();
        return false;
    }
    return true;
}

void RkMppCaptureNode::close_decoder() {
    if (decoder_) {
        mpp_destroy(decoder_);
    }
    decoder_ = nullptr;
    decoder_api_ = nullptr;
    pending_packets_.clear();
}

bool RkMppCaptureNode::open_input() {
    close_input();
    input_context_                   = avformat_alloc_context();
    if (!input_context_) {
        return false;
    }
    input_context_->interrupt_callback = AVIOInterruptCB{interrupt_io, &capture_running_};
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set_int(&options, "stimeout", static_cast<int64_t>(open_timeout_ms_) * 1000, 0);
    av_dict_set(&options, "fflags", "nobuffer", 0);
    const int open_result = avformat_open_input(&input_context_, input_.c_str(), nullptr, &options);
    av_dict_free(&options);
    if (open_result < 0 || avformat_find_stream_info(input_context_, nullptr) < 0) {
        LOG_WARN("RkMppCaptureNode {} could not open {}", getName(), input_);
        close_input();
        return false;
    }
    video_stream_index_ = av_find_best_stream(input_context_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_index_ < 0) {
        close_input();
        return false;
    }
    AVStream* stream = input_context_->streams[video_stream_index_];
    if (stream->codecpar->codec_id != AV_CODEC_ID_H264 && stream->codecpar->codec_id != AV_CODEC_ID_HEVC) {
        LOG_ERROR("RkMppCaptureNode {} only supports H.264/H.265", getName());
        close_input();
        return false;
    }
    width_            = stream->codecpar->width;
    height_           = stream->codecpar->height;
    bitrate_          = stream->codecpar->bit_rate;
    stream_time_base_ = stream->time_base;
    const AVRational rate = av_guess_frame_rate(input_context_, stream, nullptr);
    if (rate.num > 0 && rate.den > 0) {
        fps_ = std::clamp(static_cast<int>(std::round(av_q2d(rate))), 1, 240);
    }
    const char* filter_name = stream->codecpar->codec_id == AV_CODEC_ID_HEVC ? "hevc_mp4toannexb"
                                                                             : "h264_mp4toannexb";
    const AVBitStreamFilter* filter = av_bsf_get_by_name(filter_name);
    if (!filter || av_bsf_alloc(filter, &bitstream_filter_) < 0 ||
        avcodec_parameters_copy(bitstream_filter_->par_in, stream->codecpar) < 0) {
        close_input();
        return false;
    }
    bitstream_filter_->time_base_in = stream->time_base;
    if (av_bsf_init(bitstream_filter_) < 0 || !init_decoder(stream->codecpar->codec_id)) {
        close_input();
        return false;
    }
    const auto* parameters = bitstream_filter_->par_out;
    if (parameters && parameters->extradata && parameters->extradata_size > 0) {
        codec_extradata_.assign(parameters->extradata,
                                parameters->extradata + parameters->extradata_size);
    }
    LOG_INFO("RkMppCaptureNode {} opened {} {}x{} at {} fps with Rockchip MPP", getName(), input_, width_, height_,
             fps_);
    return true;
}

void RkMppCaptureNode::close_input() {
    if (bitstream_filter_) {
        av_bsf_free(&bitstream_filter_);
    }
    if (input_context_) {
        avformat_close_input(&input_context_);
    }
    video_stream_index_ = -1;
    codec_extradata_.clear();
    close_decoder();
}

object_meta::EncodedPacket::ptr RkMppCaptureNode::make_packet(const AVPacket* packet) const {
    const AVRational time_base = bitstream_filter_ ? bitstream_filter_->time_base_out : stream_time_base_;
    auto result         = std::make_shared<object_meta::EncodedPacket>();
    result->bytes.assign(packet->data, packet->data + packet->size);
    result->codec_extradata = codec_extradata_;
    result->pts_us      = to_us(packet->pts, time_base);
    result->dts_us      = to_us(packet->dts, time_base);
    result->duration_us = to_us(packet->duration, time_base);
    result->key_frame   = (packet->flags & AV_PKT_FLAG_KEY) != 0;
    result->codec       = codec_;
    return result;
}

bool RkMppCaptureNode::submit_packet(const AVPacket* packet) {
    auto encoded = make_packet(packet);
    MppPacket mpp_packet = nullptr;
    if (mpp_packet_init(&mpp_packet, encoded->bytes.data(), encoded->bytes.size()) != MPP_OK) {
        return false;
    }
    mpp_packet_set_pts(mpp_packet, encoded->pts_us);
    mpp_packet_set_dts(mpp_packet, encoded->dts_us);
    const MPP_RET result = decoder_api_->decode_put_packet(decoder_, mpp_packet);
    mpp_packet_deinit(&mpp_packet);
    if (result == MPP_OK) {
        pending_packets_.push_back(std::move(encoded));
        while (pending_packets_.size() > 128) {
            pending_packets_.pop_front();
        }
        return true;
    }
    return false;
}

void RkMppCaptureNode::drain_frames() {
    while (capture_running_) {
        MppFrame frame = nullptr;
        if (decoder_api_->decode_get_frame(decoder_, &frame) != MPP_OK || !frame) {
            break;
        }
        if (mpp_frame_get_info_change(frame)) {
            decoder_api_->control(decoder_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
            mpp_frame_deinit(&frame);
            continue;
        }
        const bool invalid = mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame);
        MppBuffer buffer = mpp_frame_get_buffer(frame);
        const int width = static_cast<int>(mpp_frame_get_width(frame));
        const int height = static_cast<int>(mpp_frame_get_height(frame));
        const int horizontal_stride = static_cast<int>(mpp_frame_get_hor_stride(frame));
        const int vertical_stride = static_cast<int>(mpp_frame_get_ver_stride(frame));
        const auto format = mpp_frame_get_fmt(frame);
        const bool linear_nv12 = (format & MPP_FRAME_FMT_MASK) == MPP_FMT_YUV420SP &&
                                 !MPP_FRAME_FMT_IS_FBC(format);
        if (!invalid && buffer && width > 0 && height > 0 && linear_nv12) {
            mpp_buffer_sync_ro_begin(buffer);
            auto* pointer = static_cast<uint8_t*>(mpp_buffer_get_ptr(buffer));
            cv::Mat yuv(vertical_stride + vertical_stride / 2, horizontal_stride, CV_8UC1, pointer);
            cv::Mat bgr;
            cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
            bgr = bgr(cv::Rect(0, 0, width, height)).clone();
            mpp_buffer_sync_ro_end(buffer);

            const int64_t decoded_pts = mpp_frame_get_pts(frame);
            auto packet_it = std::find_if(pending_packets_.begin(), pending_packets_.end(),
                                          [decoded_pts](const auto& item) { return item->pts_us == decoded_pts; });
            if (packet_it == pending_packets_.end() && decoded_pts == 0 && !pending_packets_.empty()) {
                packet_it = pending_packets_.begin();
            }
            object_meta::EncodedPacket::ptr encoded;
            if (packet_it != pending_packets_.end()) {
                encoded = *packet_it;
                pending_packets_.erase(packet_it);
            } else if (++missing_packet_count_ == 1 || missing_packet_count_ % 100 == 0) {
                LOG_WARN("RkMppCaptureNode {} has no encoded packet matching decoded PTS {}; SEI output skips this frame",
                         getName(), decoded_pts);
            }
            auto data = std::make_shared<Data::BaseData>();
            data->data_name = getName();
            data->set_frame_meta(std::make_shared<object_meta::FrameMeta>(std::move(bgr), frame_index_++));
            auto meta = std::make_shared<object_meta::PacketMeta>(fps_, width, height, static_cast<int>(bitrate_),
                                                                  codec_ == object_meta::VideoCodec::H265 ? 1 : 0);
            meta->encoded_packet = std::move(encoded);
            data->set_packet_meta(std::move(meta));
            add_data(data);
        } else if (!invalid && !linear_nv12) {
            LOG_WARN("RkMppCaptureNode {} unsupported MPP output format {}", getName(), static_cast<int>(format));
        }
        mpp_frame_deinit(&frame);
    }
}

void RkMppCaptureNode::capture_loop() {
    while (capture_running_) {
        if (!open_input()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_ms_));
            continue;
        }
        AVPacket* packet = av_packet_alloc();
        while (capture_running_ && av_read_frame(input_context_, packet) >= 0) {
            if (packet->stream_index == video_stream_index_ && av_bsf_send_packet(bitstream_filter_, packet) >= 0) {
                AVPacket* filtered = av_packet_alloc();
                while (av_bsf_receive_packet(bitstream_filter_, filtered) == 0) {
                    int attempts = 0;
                    while (capture_running_ && !submit_packet(filtered) && ++attempts < 10) {
                        drain_frames();
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                    drain_frames();
                    av_packet_unref(filtered);
                }
                av_packet_free(&filtered);
            }
            av_packet_unref(packet);
        }
        av_packet_free(&packet);
        close_input();
        if (capture_running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_ms_));
        }
    }
}

int RkMppCaptureNode::after_start() {
    capture_running_ = true;
    capture_thread_ = std::thread(&RkMppCaptureNode::capture_loop, this);
    return 0;
}

int RkMppCaptureNode::on_exit() {
    capture_running_ = false;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    close_input();
    return 0;
}

Data::BaseData::ptr RkMppCaptureNode::handle_data(Data::BaseData::ptr data) {
    return data;
}

}  // namespace Node

namespace {
Register<GraphCore::Node, Node::RkMppCaptureNode, std::string> register_rkmpp_capture("RkMppCaptureNode");
}
