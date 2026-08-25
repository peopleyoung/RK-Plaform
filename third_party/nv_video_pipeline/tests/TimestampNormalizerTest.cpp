#include <cassert>
#include <cstdint>

#include "TimestampNormalizer.h"

int main() {
    media::TimestampNormalizer normalizer;

    const auto first = normalizer.normalize(9'000'000, 9'000'000, 50'000, 20);
    assert(first.dts_us == 0);
    assert(first.pts_us == 0);
    assert(first.duration_us == 50'000);

    const auto forward = normalizer.normalize(9'050'000, 9'050'000, 50'000, 20);
    assert(forward.dts_us == 50'000);
    assert(forward.pts_us == 50'000);

    const auto backwards = normalizer.normalize(1'000, 1'000, 0, 20);
    assert(backwards.dts_us == 100'000);
    assert(backwards.pts_us == 100'000);
    assert(backwards.duration_us == 50'000);

    const auto jump = normalizer.normalize(99'000'000, 99'000'000, 50'000, 20);
    assert(jump.dts_us == 150'000);
    assert(jump.pts_us == 150'000);

    normalizer.reset();
    const auto reset = normalizer.normalize(3'000, 2'000, 0, 25);
    assert(reset.dts_us == 0);
    assert(reset.pts_us == 1'000);
    assert(reset.duration_us == 40'000);
    return 0;
}
