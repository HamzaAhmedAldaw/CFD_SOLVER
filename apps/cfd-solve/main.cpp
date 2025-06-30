// === apps/cfd-solve/main.cpp ===
#include <iostream>
#include <memory>
#include <chrono>
#include <csignal>
#include "cfd/CFDSolver.hpp"
#include "cfd/parallel/MPI_Wrapper.hpp"
#include "cfd/io/Logger.hpp"

using namespace cfd;

// Global solver pointer for signal handling
std::unique_ptr<CFDSolver> g_solver;
volatile std::sig_atomic_t g_signalReceived = 0;

void signalHandler(int signal) {
    g_signalReceived = signal;
    if (g_solver) {
        // Trigger graceful shutdown
        // g_solver->stop();
    }
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options] <case_directory>\n"
              << "\nOptions:\n"
              << "  -h, --help           Show this help message\n"
              << "  -v, --verbose        Enable verbose output\n"
              << "  -p, --parallel       Run in parallel (MPI must be initialized)\n"
              << "  -r, --restart <time> Restart from specified time\n"
              << "  -e, --end-time <t>   Override end time\n"
              << "  -d, --delta-t <dt>   Override time step\n"
              << "  -w, --write <int>    Override write interval\n"
              << "\nSignals:\n"
              << "  SIGINT  (Ctrl+C)     Clean shutdown\n"
              << "  SIGUSR1              Write current fields\n"
              << "  SIGUSR2              Print solver statistics\n";
}

struct SolverOptions {
    std::string caseDir;
    bool verbose = false;
    bool parallel = false;
    Real restartTime = -1.0;
    Real endTime = -1.0;
    Real deltaT = -1.0;
    Real writeInterval = -1.0;
};

SolverOptions parseArguments(int argc, char* argv[]) {
    SolverOptions opts;
    
    if (argc < 2) {
        printUsage(argv[0]);
        std::exit(1);
    }
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "-p" || arg == "--parallel") {
            opts.parallel = true;
        } else if (arg == "-r" || arg == "--restart") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -r requires an argument\n";
                std::exit(1);
            }
            opts.restartTime = std::stod(argv[++i]);
        } else if (arg == "-e" || arg == "--end-time") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -e requires an argument\n";
                std::exit(1);
            }
            opts.endTime = std::stod(argv[++i]);
        } else if (arg == "-d" || arg == "--delta-t") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -d requires an argument\n";
                std::exit(1);
            }
            opts.deltaT = std::stod(argv[++i]);
        } else if (arg == "-w" || arg == "--write") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -w requires an argument\n";
                std::exit(1);
            }
            opts.writeInterval = std::stod(argv[++i]);
        } else if (arg[0] != '-') {
            opts.caseDir = arg;
        } else {
            std::cerr << "Error: Unknown option " << arg << "\n";
            printUsage(argv[0]);
            std::exit(1);
        }
    }
    
    if (opts.caseDir.empty()) {
        std::cerr << "Error: No case directory specified\n";
        printUsage(argv[0]);
        std::exit(1);
    }
    
    return opts;
}

int main(int argc, char* argv[]) {
    // Parse command line
    SolverOptions opts = parseArguments(argc, argv);
    
    // Initialize MPI if requested
    if (opts.parallel) {
        parallel::MPIWrapper::initialize(argc, argv);
    }
    
    // Setup signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::signal(SIGUSR1, signalHandler);
    std::signal(SIGUSR2, signalHandler);
    
    try {
        // Setup logger
        auto logger = io::Logger::instance();
        if (opts.verbose) {
            logger.setLevel(io::Logger::Level::DEBUG);
        }
        
        // Print startup info
        logger.info("========================================");
        logger.info("CFD Solver v1.0.0");
        logger.info("========================================");
        
        if (parallel::MPIWrapper::isParallel()) {
            logger.info("Running in parallel on {} processors",
                       parallel::MPIWrapper::size());
            logger.info("This is rank {}", parallel::MPIWrapper::rank());
        }
        
        // Start timing
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Create and run solver
        g_solver = std::make_unique<CFDSolver>(opts.caseDir);
        
        // Override settings if specified
        if (opts.restartTime >= 0) {
            // Implement restart functionality
            logger.info("Restarting from time = {}", opts.restartTime);
        }
        
        // Run simulation
        g_solver->run();
        
        // Compute elapsed time
        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            endTime - startTime);
        
        logger.info("========================================");
        logger.info("Simulation completed successfully");
        logger.info("Total elapsed time: {} seconds", elapsed.count());
        logger.info("========================================");
        
        // Cleanup
        g_solver.reset();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        
        if (parallel::MPIWrapper::isParallel()) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return 1;
    }
    
    // Finalize MPI
    if (opts.parallel) {
        parallel::MPIWrapper::finalize();
    }
    
    return 0;
}
