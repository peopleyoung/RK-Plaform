#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace toolkit {

inline uint64_t getCurrentMillisecond(bool = false) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
}

inline std::string str_format(const char* format, ...) {
    va_list args;
    va_start(args, format);
    va_list copy;
    va_copy(copy, args);
    const int size = std::vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    if (size < 0) {
        va_end(args);
        return {};
    }
    std::vector<char> buffer(static_cast<size_t>(size) + 1);
    std::vsnprintf(buffer.data(), buffer.size(), format, args);
    va_end(args);
    return std::string(buffer.data(), static_cast<size_t>(size));
}

inline std::vector<std::string> split(const std::string& value, const std::string& delimiter) {
    std::vector<std::string> result;
    if (delimiter.empty()) {
        result.emplace_back(value);
        return result;
    }
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t end = value.find(delimiter, begin);
        result.emplace_back(value.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + delimiter.size();
    }
    return result;
}

inline bool start_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

inline void replace(std::string& value, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return;
    }
    size_t position = 0;
    while ((position = value.find(from, position)) != std::string::npos) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}

inline std::string exePath() {
    std::error_code error;
    const auto      path = std::filesystem::read_symlink("/proc/self/exe", error);
    return error ? std::string("video_pipeline") : path.string();
}

}  // namespace toolkit
