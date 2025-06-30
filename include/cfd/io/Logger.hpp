#pragma once

#include "cfd/core/Types.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/ostr.h>
#include <memory>
#include <string>
#include <chrono>

namespace cfd::io {

// Logger wrapper class
class Logger {
public:
    enum class Level {
        TRACE = SPDLOG_LEVEL_TRACE,
        DEBUG = SPDLOG_LEVEL_DEBUG,
        INFO = SPDLOG_LEVEL_INFO,
        WARN = SPDLOG_LEVEL_WARN,
        ERROR = SPDLOG_LEVEL_ERROR,
        CRITICAL = SPDLOG_LEVEL_CRITICAL
    };
    
    // Get singleton instance
    static Logger* getInstance() {
        static Logger instance;
        return &instance;
    }
    
    // Initialize logger
    void initialize(const std::string& logFile = "cfd_solver.log",
                   Level consoleLevel = Level::INFO,
                   Level fileLevel = Level::DEBUG) {
        try {
            // Console sink
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(static_cast<spdlog::level::level_enum>(consoleLevel));
            console_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
            
            // File sink
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFile, true);
            file_sink->set_level(static_cast<spdlog::level::level_enum>(fileLevel));
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
            
            // Create logger with both sinks
            std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
            logger_ = std::make_shared<spdlog::logger>("cfd", sinks.begin(), sinks.end());
            logger_->set_level(spdlog::level::trace);
            
            // Register logger
            spdlog::register_logger(logger_);
            spdlog::set_default_logger(logger_);
            
            // Set flush policy
            logger_->flush_on(spdlog::level::warn);
            spdlog::flush_every(std::chrono::seconds(5));
            
        } catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        }
    }
    
    // Logging functions
    template<typename... Args>
    void trace(const std::string& fmt, Args&&... args) {
        logger_->trace(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void debug(const std::string& fmt, Args&&... args) {
        logger_->debug(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void info(const std::string& fmt, Args&&... args) {
        logger_->info(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void warn(const std::string& fmt, Args&&... args) {
        logger_->warn(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void error(const std::string& fmt, Args&&... args) {
        logger_->error(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void critical(const std::string& fmt, Args&&... args) {
        logger_->critical(fmt, std::forward<Args>(args)...);
    }
    
    // Set log level
    void setLevel(Level level) {
        logger_->set_level(static_cast<spdlog::level::level_enum>(level));
    }
    
    // Flush logger
    void flush() {
        logger_->flush();
    }
    
    // Performance timer
    class Timer {
    public:
        Timer(const std::string& name, Logger* logger = getInstance())
            : name_(name), logger_(logger), start_(std::chrono::high_resolution_clock::now()) {
            logger_->debug("Timer '{}' started", name_);
        }
        
        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            logger_->info("Timer '{}' elapsed: {:.3f} ms", name_, duration.count() / 1000.0);
        }
        
        Real elapsed() const {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_);
            return duration.count() / 1.0e6;  // Return seconds
        }
        
    private:
        std::string name_;
        Logger* logger_;
        std::chrono::high_resolution_clock::time_point start_;
    };
    
    // Create timer
    std::unique_ptr<Timer> createTimer(const std::string& name) {
        return std::make_unique<Timer>(name, this);
    }
    
    // Progress indicator
    class Progress {
    public:
        Progress(const std::string& task, int total, Logger* logger = getInstance())
            : task_(task), total_(total), current_(0), logger_(logger),
              lastUpdate_(std::chrono::steady_clock::now()) {
            logger_->info("{} started (0/{})", task_, total_);
        }
        
        void update(int current) {
            current_ = current;
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate_);
            
            // Update every 500ms or at completion
            if (elapsed.count() > 500 || current_ >= total_) {
                int percent = (current_ * 100) / total_;
                logger_->info("{}: {}% ({}/{})", task_, percent, current_, total_);
                lastUpdate_ = now;
            }
        }
        
        void increment() {
            update(current_ + 1);
        }
        
        void finish() {
            if (current_ < total_) {
                update(total_);
            }
            logger_->info("{} completed", task_);
        }
        
    private:
        std::string task_;
        int total_;
        int current_;
        Logger* logger_;
        std::chrono::steady_clock::time_point lastUpdate_;
    };
    
    // Create progress indicator
    std::unique_ptr<Progress> createProgress(const std::string& task, int total) {
        return std::make_unique<Progress>(task, total, this);
    }
    
    // Memory usage logging
    void logMemoryUsage(const std::string& context = "") {
        // Platform-specific memory usage retrieval
        #ifdef __linux__
        std::ifstream status("/proc/self/status");
        std::string line;
        size_t vmSize = 0, vmRSS = 0;
        
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmSize") {
                vmSize = std::stoul(line.substr(7));
            } else if (line.substr(0, 5) == "VmRSS") {
                vmRSS = std::stoul(line.substr(6));
            }
        }
        
        info("Memory usage{}: Virtual={:.1f} MB, Resident={:.1f} MB",
             context.empty() ? "" : " (" + context + ")",
             vmSize / 1024.0, vmRSS / 1024.0);
        #else
        info("Memory usage logging not implemented for this platform");
        #endif
    }
    
    // Residual logging
    void logResidual(const std::string& field, int iteration, Real residual,
                    Real initialResidual = 1.0) {
        Real relResidual = residual / (initialResidual + SMALL);
        info("Iteration {:4d}: {} residual = {:.6e} (rel = {:.6e})",
             iteration, field, residual, relResidual);
    }
    
    // Convergence summary
    void logConvergence(const std::map<std::string, Real>& residuals,
                       const std::map<std::string, Real>& initialResiduals) {
        info("Convergence summary:");
        for (const auto& [field, residual] : residuals) {
            Real initial = initialResiduals.count(field) ? initialResiduals.at(field) : 1.0;
            Real relative = residual / (initial + SMALL);
            info("  {:<10s}: {:.6e} (rel = {:.6e})", field, residual, relative);
        }
    }
    
    // Performance metrics
    struct PerformanceMetrics {
        Real totalTime = 0.0;
        Real solverTime = 0.0;
        Real ioTime = 0.0;
        Real communicationTime = 0.0;
        int iterations = 0;
        size_t memoryUsage = 0;
    };
    
    void logPerformance(const PerformanceMetrics& metrics) {
        info("Performance Summary:");
        info("  Total time:         {:.2f} s", metrics.totalTime);
        info("  Solver time:        {:.2f} s ({:.1f}%)", 
             metrics.solverTime, 100.0 * metrics.solverTime / metrics.totalTime);
        info("  I/O time:           {:.2f} s ({:.1f}%)", 
             metrics.ioTime, 100.0 * metrics.ioTime / metrics.totalTime);
        info("  Communication time: {:.2f} s ({:.1f}%)", 
             metrics.communicationTime, 100.0 * metrics.communicationTime / metrics.totalTime);
        info("  Iterations:         {}", metrics.iterations);
        info("  Time per iteration: {:.3f} s", metrics.totalTime / metrics.iterations);
        info("  Memory usage:       {:.1f} MB", metrics.memoryUsage / (1024.0 * 1024.0));
    }
    
private:
    std::shared_ptr<spdlog::logger> logger_;
    
    Logger() {
        // Initialize with default settings
        initialize();
    }
    
    // Delete copy constructor and assignment
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};

// Convenience macros
#define LOG_TRACE(...) cfd::io::Logger::getInstance()->trace(__VA_ARGS__)
#define LOG_DEBUG(...) cfd::io::Logger::getInstance()->debug(__VA_ARGS__)
#define LOG_INFO(...) cfd::io::Logger::getInstance()->info(__VA_ARGS__)
#define LOG_WARN(...) cfd::io::Logger::getInstance()->warn(__VA_ARGS__)
#define LOG_ERROR(...) cfd::io::Logger::getInstance()->error(__VA_ARGS__)
#define LOG_CRITICAL(...) cfd::io::Logger::getInstance()->critical(__VA_ARGS__)

#define LOG_TIMER(name) auto _timer_##__LINE__ = cfd::io::Logger::getInstance()->createTimer(name)
#define LOG_PROGRESS(task, total) auto _progress_##__LINE__ = cfd::io::Logger::getInstance()->createProgress(task, total)

// Stream-style logging support
class LogStream {
public:
    LogStream(Logger::Level level) : level_(level) {}
    
    ~LogStream() {
        auto logger = Logger::getInstance();
        switch (level_) {
            case Logger::Level::TRACE: logger->trace(stream_.str()); break;
            case Logger::Level::DEBUG: logger->debug(stream_.str()); break;
            case Logger::Level::INFO: logger->info(stream_.str()); break;
            case Logger::Level::WARN: logger->warn(stream_.str()); break;
            case Logger::Level::ERROR: logger->error(stream_.str()); break;
            case Logger::Level::CRITICAL: logger->critical(stream_.str()); break;
        }
    }
    
    template<typename T>
    LogStream& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }
    
private:
    Logger::Level level_;
    std::ostringstream stream_;
};

#define LOG(level) cfd::io::LogStream(cfd::io::Logger::Level::level)

} // namespace cfd::io