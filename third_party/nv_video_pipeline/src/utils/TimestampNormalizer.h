#pragma once

#include <algorithm>
#include <cstdint>

namespace media {

struct NormalizedTimestamp {
    int64_t pts_us{0};
    int64_t dts_us{0};
    int64_t duration_us{0};
};

class TimestampNormalizer {
public:
    NormalizedTimestamp normalize(int64_t source_pts_us, int64_t source_dts_us,
                                  int64_t source_duration_us, int fps) {
        const int64_t nominal_duration = 1'000'000 / std::clamp(fps, 1, 240);
        const int64_t duration =
            source_duration_us > 0 && source_duration_us <= 1'000'000
                ? source_duration_us
                : nominal_duration;
        if (!initialized_) {
            initialized_ = true;
            last_source_dts_us_ = source_dts_us;
            last_output_dts_us_ = 0;
            return {pts_offset(source_pts_us, source_dts_us, duration), 0, duration};
        }

        const int64_t source_delta = source_dts_us - last_source_dts_us_;
        const int64_t maximum_delta = std::max<int64_t>(1'000'000, duration * 10);
        const int64_t step = source_dts_us > 0 && last_source_dts_us_ > 0
                                     && source_delta > 0 && source_delta <= maximum_delta
                                 ? source_delta
                                 : duration;
        last_source_dts_us_ = source_dts_us;
        last_output_dts_us_ += step;
        return {
            last_output_dts_us_ + pts_offset(source_pts_us, source_dts_us, duration),
            last_output_dts_us_,
            duration,
        };
    }

    void reset() {
        initialized_ = false;
        last_source_dts_us_ = 0;
        last_output_dts_us_ = 0;
    }

private:
    static int64_t pts_offset(int64_t pts_us, int64_t dts_us, int64_t duration_us) {
        const int64_t offset = pts_us - dts_us;
        const int64_t maximum_offset = std::max<int64_t>(1'000'000, duration_us * 4);
        return offset >= 0 && offset <= maximum_offset ? offset : 0;
    }

    bool initialized_{false};
    int64_t last_source_dts_us_{0};
    int64_t last_output_dts_us_{0};
};

}  // namespace media
