#include "Logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/async.h>
#include <chrono>
#include <mutex>

namespace common {

static std::mutex g_loggerMtxLock; // ensure timestamp consistency

Logger::Logger(const std::string& logger_id, const std::string& log_file_path, const std::string& async_log_file_path){
    
    {
        std::lock_guard<std::mutex> lock(g_loggerMtxLock);
        auto ts = std::chrono::system_clock::now().time_since_epoch().count();
        auto logger_name = logger_id + "_" + std::to_string(ts);
        m_stdout_logger = spdlog::stdout_color_mt(logger_name);
        ts = std::chrono::system_clock::now().time_since_epoch().count();
        logger_name = logger_id + "_" + std::to_string(ts);
        m_file_logger = spdlog::basic_logger_mt(logger_name, log_file_path);
        ts = std::chrono::system_clock::now().time_since_epoch().count();
        logger_name = logger_id + "_" + std::to_string(ts);
        m_async_file_logger = spdlog::basic_logger_mt<spdlog::async_factory>(logger_name, async_log_file_path);
    }

    // Initialize the default output levels of each logger here
    setLoggersPrintLevel(Loglevel::Debug, Loglevel::Info, Loglevel::Info);

}

Logger::~Logger() {
    // Cleanup the logger here
    spdlog::flush_every(std::chrono::seconds(3));
    spdlog::shutdown();
    m_stdout_logger.reset();
    m_file_logger.reset();
    m_async_file_logger.reset();
}

} // namespace common
