#include "SeiPacket.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr std::array<uint8_t, 16> kExpectedUuid = {
    0x94, 0x51, 0xef, 0x8f, 0xd2, 0x41, 0x49, 0x6a,
    0x80, 0xba, 0x68, 0x18, 0xe2, 0x4d, 0xc0, 0x4e,
};

std::vector<uint8_t> rbsp_from_packet(const std::vector<uint8_t>& packet,
                                      object_meta::VideoCodec codec) {
    assert(packet.size() > 8);
    assert((packet[0] == 0 && packet[1] == 0 && packet[2] == 0 && packet[3] == 1));
    size_t offset = 4;
    if (codec == object_meta::VideoCodec::H265) {
        assert(packet[offset++] == 0x4e);
        assert(packet[offset++] == 0x01);
    } else {
        assert(packet[offset++] == 0x06);
    }
    std::vector<uint8_t> rbsp;
    int zero_count = 0;
    for (; offset < packet.size(); ++offset) {
        const uint8_t byte = packet[offset];
        if (zero_count >= 2 && byte == 0x03) {
            zero_count = 0;
            continue;
        }
        rbsp.push_back(byte);
        zero_count = byte == 0 ? zero_count + 1 : 0;
    }
    return rbsp;
}

void verify_payload(object_meta::VideoCodec codec) {
    std::string payload(300, 'x');
    payload[10] = '\0';
    payload[11] = '\0';
    payload[12] = '\1';
    const auto packet = media::make_user_data_sei(payload, codec);
    assert(packet.has_value());
    const auto rbsp = rbsp_from_packet(*packet, codec);
    assert(rbsp[0] == 5);
    assert(rbsp[1] == 0xff);
    assert(rbsp[2] == 61);
    assert(std::equal(kExpectedUuid.begin(), kExpectedUuid.end(), rbsp.begin() + 3));
    assert(std::equal(payload.begin(), payload.end(), rbsp.begin() + 19));
    assert(rbsp.back() == 0x80);

    bool found_prevention = false;
    for (size_t index = codec == object_meta::VideoCodec::H265 ? 6 : 5;
         index + 3 < packet->size(); ++index) {
        if ((*packet)[index] == 0 && (*packet)[index + 1] == 0
            && (*packet)[index + 2] == 3 && (*packet)[index + 3] == 1) {
            found_prevention = true;
        }
    }
    assert(found_prevention);
}

}  // namespace

int main() {
    verify_payload(object_meta::VideoCodec::H264);
    verify_payload(object_meta::VideoCodec::H265);

    std::string maximum(media::kMaxSeiUserPayloadBytes, 'a');
    assert(media::make_user_data_sei(maximum, object_meta::VideoCodec::H264).has_value());
    maximum.push_back('b');
    assert(!media::make_user_data_sei(maximum, object_meta::VideoCodec::H264).has_value());

    assert(media::sei_metadata_within_limits(3840, 2160, 262144));
    assert(!media::sei_metadata_within_limits(3841, 2160, 1));
    assert(!media::sei_metadata_within_limits(3840, 2161, 1));
    assert(!media::sei_metadata_within_limits(3840, 2160, 262145));
    return 0;
}
