#include "CLog.h"

#include <Util/util.h>
#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <iostream>
#include <string>

using namespace std;

Clog::Clog() {
}

Clog::~Clog() {
}

Clog* Clog::getInstance() {
    static Clog clogger;
    return &clogger;
}

std::shared_ptr<spdlog::logger> Clog::getLogger() {
    return m_pLogger;
}

bool Clog::Init(const LogConfig& config) {
    m_config = config;
    try {
        // sink容器
        std::vector<spdlog::sink_ptr> vecSink;

        // 控制台
        if (m_config.out_pos & OutPosition::CONSOLE_) {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern(LOG_OUTPUT_FORMAT);
            vecSink.push_back(console_sink);
        }

        // 文件
        if (m_config.out_pos & OutPosition::FILE_) {
            auto file_name = (toolkit::exePath() + ".log");
            auto file_sink =
                std::make_shared<spdlog::sinks::rotating_file_sink_mt>(file_name, m_config.max_size, m_config.max_file);
            file_sink->set_pattern(LOG_OUTPUT_FORMAT);
            vecSink.push_back(file_sink);
        }

        // 设置logger使用多个sink
        if (m_config.out_mode == ASYNC)  // 异步
        {
            spdlog::init_thread_pool(102400, 1);
            auto tp   = spdlog::thread_pool();
            m_pLogger = std::make_shared<spdlog::async_logger>(LOG_NAME, begin(vecSink), end(vecSink), tp,
                                                               spdlog::async_overflow_policy::block);
        } else  // 同步
        {
            m_pLogger = std::make_shared<spdlog::logger>(LOG_NAME, begin(vecSink), end(vecSink));
        }
        m_pLogger->set_level(m_config.out_level);

        // 遇到warn级别，立即flush到文件
        m_pLogger->flush_on(spdlog::level::warn);
        // 定时flush到文件，每三秒刷新一次
        spdlog::flush_every(std::chrono::seconds(3));
        spdlog::register_logger(m_pLogger);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cout << "Log initialization failed: " << ex.what() << std::endl;
        return false;
    }
    return true;
}

void Clog::UnInit() {
    spdlog::drop_all();
    spdlog::shutdown();
}
