#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include <string>
#include <iostream>
#include <memory>
#include <string_view>
#include <chrono>
#include <mutex>
#include <spdlog/spdlog.h>

namespace common {

using namespace std::string_literals;

class Logger {
public:
    Logger(const std::string& logger_id, const std::string& log_file_path = "logs/project.log"s, const std::string& async_log_file_path = "logs/project_async.log"s);
    virtual ~Logger();

    enum class LogLevel {
        Debug=1,    // synchronize with spdlog::level
        Info,
        Warn,
        Error,
        Critical
    };
    Logger(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger& operator=(Logger&&) = delete;

    void setLoggersPrintLevel(LogLevel stdout_level, LogLevel file_level, LogLevel async_file_level){
        auto spd_level = static_cast<spdlog::level::level_enum>(stdout_level);
        m_stdout_logger->set_level(spd_level);
        spd_level = static_cast<spdlog::level::level_enum>(file_level);
        m_file_logger->set_level(spd_level);
        m_file_logger->flush_on(spd_level);
        spd_level = static_cast<spdlog::level::level_enum>(async_file_level);
        m_async_file_logger->set_level(spd_level);
        m_async_file_logger->flush_on(spd_level);
    }

    void setPattern(const std::string& pattern = "[%H:%M:%S.%f][%^%l%$] %v"s){
        // Set the pattern for the loggers
        spdlog::set_pattern(pattern);
    }

    template<typename... Args>
    void printStdoutLog(LogLevel level, std::string_view sv, const Args &... args){
        // Print the log message to stdout
        printLogger(m_stdout_logger, level, sv, args...);
    }

    template<typename... Args>
    void printFileLog(LogLevel level, std::string_view sv, const Args &... args){
        // Print the log message to a file
        printLogger(m_file_logger, level, sv, args...);
    }


    template<typename... Args>
    void printAsyncFileLog(LogLevel level, std::string_view sv, const Args &... args){
        // Print the log message to a file asynchronously
        printLogger(m_async_file_logger, level, sv, args...);
    }

protected:
    template<typename... Args>
    void printLogger(std::shared_ptr<spdlog::logger> &logger, LogLevel level, std::string_view sv, const Args &... args){
        // Print the log message to the logger
        switch (level)
        {
        case LogLevel::Debug:
            logger->debug(sv, args...);
            break;
        case LogLevel::Info:
            logger->info(sv, args...);
            break;
        case LogLevel::Warn:
            logger->warn(sv, args...);
            break;
        case LogLevel::Error:
            logger->error(sv, args...);
            break;
        case LogLevel::Critical:
            logger->critical(sv, args...);
            break;
        default:
            break;
        }
    }

private:
    static std::mutex m_loggerMtxLock; // ensure timestamp consistency

    std::shared_ptr<spdlog::logger> m_stdout_logger{nullptr};
    std::shared_ptr<spdlog::logger> m_file_logger{nullptr};
    std::shared_ptr<spdlog::logger> m_async_file_logger{nullptr};
};


} // namespace common

#endif // __LOGGER_HPP__
