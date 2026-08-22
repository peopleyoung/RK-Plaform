#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace object_meta {
enum class VideoCodec {
    H264 = 0,
    H265 = 1,
};

struct EncodedPacket {
    using ptr = std::shared_ptr<EncodedPacket>;

    std::vector<uint8_t> bytes;
    std::vector<uint8_t> codec_extradata;
    int64_t              pts_us{0};
    int64_t              dts_us{0};
    int64_t              duration_us{0};
    bool                 key_frame{false};
    VideoCodec           codec{VideoCodec::H264};
};

class PacketMeta {
public:
    using ptr = std::shared_ptr<PacketMeta>;

    PacketMeta(int fps = 0, int width = 0, int height = 0, int bitrate = 0, int codec_id = 0)
        : fps(fps), width(width), height(height), bitrate(bitrate), codec_id(codec_id) {
    }

    int fps      = 0;
    int width    = 0;
    int height   = 0;
    int bitrate  = 0;
    int codec_id = 0;
    EncodedPacket::ptr encoded_packet;
};
}  // namespace object_meta
