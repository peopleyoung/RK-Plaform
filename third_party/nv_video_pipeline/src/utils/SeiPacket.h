#pragma once

#include <cstdint>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "PacketMeta.h"

namespace media {

constexpr size_t kMaxSeiUserPayloadBytes = 1024 * 1024;
constexpr size_t kMaxSegmentationRuns = 262144;
constexpr int kMaxSegmentationWidth = 3840;
constexpr int kMaxSegmentationHeight = 2160;

std::optional<std::vector<uint8_t>> make_user_data_sei(
    const std::string& payload, object_meta::VideoCodec codec);

bool sei_metadata_within_limits(int source_width, int source_height,
                                size_t segmentation_runs);

}  // namespace media
