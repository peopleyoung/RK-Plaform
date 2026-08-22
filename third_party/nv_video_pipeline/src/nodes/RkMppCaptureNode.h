#pragma once

extern "C" {
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
}
#include <rk_mpi.h>

#include <atomic>
#include <deque>
#include <string>
#include <thread>

#include "PacketMeta.h"
#include "ProcessNode.h"

namespace Node {

class RkMppCaptureNode : public GraphCore::Node {
public:
    explicit RkMppCaptureNode(const std::string& name);
    ~RkMppCaptureNode();

    bool Init(YAML::Node config) override;

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
    int                 after_start();
    int                 on_exit();
    void                capture_loop();
    bool                open_input();
    void                close_input();
    bool                init_decoder(AVCodecID codec);
    void                close_decoder();
    bool                submit_packet(const AVPacket* packet);
    void                drain_frames();
    object_meta::EncodedPacket::ptr make_packet(const AVPacket* packet) const;

    std::string      input_;
    int              reconnect_ms_{1000};
    int              open_timeout_ms_{5000};
    std::thread      capture_thread_;
    std::atomic_bool capture_running_{false};
    AVFormatContext* input_context_{nullptr};
    AVBSFContext*    bitstream_filter_{nullptr};
    int              video_stream_index_{-1};
    AVRational       stream_time_base_{1, 1000000};
    int              fps_{25};
    int              width_{0};
    int              height_{0};
    int64_t          bitrate_{0};
    MppCtx           decoder_{nullptr};
    MppApi*          decoder_api_{nullptr};
    object_meta::VideoCodec codec_{object_meta::VideoCodec::H264};
    std::vector<uint8_t> codec_extradata_;
    std::deque<object_meta::EncodedPacket::ptr> pending_packets_;
    uint32_t         frame_index_{0};
    uint64_t         missing_packet_count_{0};
};

}  // namespace Node
