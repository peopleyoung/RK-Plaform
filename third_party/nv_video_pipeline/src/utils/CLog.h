#pragma once

#include <spdlog/common.h>

#include <string>

#include "spdlog/spdlog.h"

enum OutPosition {
    CONSOLE_          = 0x01,  // 控制台
    FILE_             = 0X02,  // 文件
    CONSOLE_AND_FILE_ = 0x03,  // 控制台+文件
};

enum OutMode {
    SYNC,   // 同步模式
    ASYNC,  // 异步模式
};

struct LogConfig {
    const char*               name      = "logger";
    spdlog::level::level_enum out_level = spdlog::level::trace;
    OutMode                   out_mode  = ASYNC;
    OutPosition               out_pos   = CONSOLE_AND_FILE_;
    int                       max_size  = 1024 * 1024 * 10;
    int                       max_file  = 10;
};

// 日志名称
#define LOG_NAME          "logger"
#define LOG_OUTPUT_FORMAT "%^[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] | %v%$"
// #define LOG_OUT_LEVEL     spdlog::level::trace  // 日志等级
// #define LOG_MAX_SIZE      1024 * 1024 * 10      // 日志文件最大大小
// #define LOG_MAX_FILE      10                    // 日志文件最大个数
// #define LOG_OUT_MODE      ASYNC                 // 日志输出模式
// #define LOG_OUT_POS       CONSOLE_AND_FILE_     // 日志输出位置
// 封装宏，没有该宏无法输出文件名、行号等信息
#define LOG_TRACE(...) SPDLOG_LOGGER_CALL(Clog::getInstance()->getLogger(), spdlog::level::trace, __VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_LOGGER_CALL(Clog::getInstance()->getLogger(), spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(...)  SPDLOG_LOGGER_CALL(Clog::getInstance()->getLogger(), spdlog::level::info, __VA_ARGS__)
#define LOG_WARN(...)  SPDLOG_LOGGER_CALL(Clog::getInstance()->getLogger(), spdlog::level::warn, __VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_LOGGER_CALL(Clog::getInstance()->getLogger(), spdlog::level::err, __VA_ARGS__)
#define LOG_CRITI(...) SPDLOG_LOGGER_CALL(Clog::getInstance()->getLogger(), spdlog::level::critical, __VA_ARGS__)

class Clog {
public:
    static Clog*                    getInstance();
    std::shared_ptr<spdlog::logger> getLogger();
    // 日志输出位置

public:
    Clog();
    ~Clog();

    /* func: 初始化日志通道
     * @para[in] nFileName    : 日志存储路径			（支持相对路径和绝对路径）
     * @para[in] nMaxFileSize : 日志文件最大存储大小	（默认1024*1024*10）
     * @para[in] nMaxFile     : 最多存储多少个日志文件	（默认10，超过最大值则循环覆盖）
     * @para[in] outMode      : 日志输出模式			（同步、异步）
     * @para[in] outPos       : 日志输出位置			（控制台、文件、控制台+文件）
     * @para[in] outLevel     : 日志输出等级			（只输出>=等级的日志消息）
     */
    // bool Init(const char* nFileName, const int nMaxFileSize = 1024 * 1024 * 10, const int nMaxFile = 10,
    //           const OutMode outMode = ASYNC, const OutPosition outPos = CONSOLE_AND_FILE_,
    //           spdlog::level::level_enum outLevel = spdlog::level::trace);
    bool Init(const LogConfig& config);
    void UnInit();

public:
    std::shared_ptr<spdlog::logger> m_pLogger;
    LogConfig                       m_config;
};
