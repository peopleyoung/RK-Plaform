#pragma once

#include <chrono>

namespace toolkit {

class Ticker {
public:
    Ticker() {
        resetTime();
    }

    void resetTime() {
        begin_ = Clock::now();
    }

    double elapsedTime() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point begin_;
};

}  // namespace toolkit
