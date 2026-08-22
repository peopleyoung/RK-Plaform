#include "SeiPacket.h"

#include <array>

namespace media {
namespace {
constexpr std::array<uint8_t, 16> kUuid = {0x94, 0x51, 0xef, 0x8f, 0xd2, 0x41, 0x49, 0x6a,
                                            0x80, 0xba, 0x68, 0x18, 0xe2, 0x4d, 0xc0, 0x4e};

std::vector<uint8_t> rbsp_to_ebsp(const std::vector<uint8_t>& rbsp) {
    std::vector<uint8_t> ebsp;
    ebsp.reserve(rbsp.size() + rbsp.size() / 32);
    size_t zero_count = 0;
    for (const uint8_t byte : rbsp) {
        if (zero_count >= 2 && byte <= 0x03) {
            ebsp.push_back(0x03);
            zero_count = 0;
        }
        ebsp.push_back(byte);
        zero_count = byte == 0 ? zero_count + 1 : 0;
    }
    return ebsp;
}
}

std::optional<std::vector<uint8_t>> make_user_data_sei(
    const std::string& payload, object_meta::VideoCodec codec) {
    if (payload.size() > kMaxSeiUserPayloadBytes) {
        return std::nullopt;
    }
    const size_t payload_size = kUuid.size() + payload.size();
    std::vector<uint8_t> rbsp;
    rbsp.reserve(payload_size + 8);
    rbsp.push_back(0x05);
    size_t remaining = payload_size;
    while (remaining >= 0xff) {
        rbsp.push_back(0xff);
        remaining -= 0xff;
    }
    rbsp.push_back(static_cast<uint8_t>(remaining));
    rbsp.insert(rbsp.end(), kUuid.begin(), kUuid.end());
    rbsp.insert(rbsp.end(), payload.begin(), payload.end());
    rbsp.push_back(0x80);

    const auto ebsp = rbsp_to_ebsp(rbsp);
    std::vector<uint8_t> result = {0x00, 0x00, 0x00, 0x01};
    if (codec == object_meta::VideoCodec::H265) {
        result.push_back(0x4e);
        result.push_back(0x01);
    } else {
        result.push_back(0x06);
    }
    result.insert(result.end(), ebsp.begin(), ebsp.end());
    return result;
}

bool sei_metadata_within_limits(int source_width, int source_height,
                                size_t segmentation_runs) {
    return source_width > 0 && source_height > 0
           && source_width <= kMaxSegmentationWidth
           && source_height <= kMaxSegmentationHeight
           && segmentation_runs <= kMaxSegmentationRuns;
}

}  // namespace media
