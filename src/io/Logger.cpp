// === src/io/Logger.cpp ===
#include "cfd/io/Logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace cfd::io {

Logger::Logger(const std::string& name) {
    // Create console sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
    
    // Create file sink
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        "cfd_solver.log", true);
    file_sink->set_level(spdlog::level::debug);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");
    
    // Create logger with both sinks
    std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
    logger_ = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
    
    // Register logger
    spdlog::register_logger(logger_);
    
    // Set default level
    logger_->set_level(spdlog::level::info);
}

Logger::~Logger() {
    spdlog::drop(logger_->name());
}

void Logger::setLevel(Level level) {
    switch (level) {
        case Level::TRACE:
            logger_->set_level(spdlog::level::trace);
            break;
        case Level::DEBUG:
            logger_->set_level(spdlog::level::debug);
            break;
        case Level::INFO:
            logger_->set_level(spdlog::level::info);
            break;
        case Level::WARN:
            logger_->set_level(spdlog::level::warn);
            break;
        case Level::ERROR:
            logger_->set_level(spdlog::level::err);
            break;
        case Level::CRITICAL:
            logger_->set_level(spdlog::level::critical);
            break;
    }
}

Logger& Logger::instance() {
    static Logger instance("global");
    return instance;
}
