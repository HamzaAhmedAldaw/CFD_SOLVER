// === apps/cfd-preprocess/main.cpp ===
#include <iostream>
#include <string>
#include <memory>
#include <cstdlib>
#include "cfd/core/Mesh.hpp"
#include "cfd/io/CaseReader.hpp"
#include "cfd/io/Logger.hpp"
#include "cfd/parallel/Domain.hpp"
#include "cfd/parallel/MPI_Wrapper.hpp"
#include <yaml-cpp/yaml.h>

using namespace cfd;

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options] <case_directory>\n"
              << "\nOptions:\n"
              << "  -h, --help                Show this help message\n"
              << "  -v, --verbose             Enable verbose output\n"
              << "  -p, --partitions <n>      Number of partitions for decomposition\n"
              << "  -m, --method <method>     Decomposition method (simple|rcb|graph|metis)\n"
              << "  -o, --output <dir>        Output directory (default: case_directory/processor*)\n"
              << "  -c, --check               Check mesh quality\n"
              << "  -s, --scale <factor>      Scale mesh by factor\n"
              << "  -r, --renumber            Renumber cells for cache efficiency\n"
              << "  -b, --balance             Show load balance statistics\n"
              << "\nExamples:\n"
              << "  " << programName << " cavity\n"
              << "  " << programName << " -p 4 -m rcb channel\n"
              << "  " << programName << " -c -v airfoil\n";
}

struct PreprocessOptions {
    std::string caseDir;
    std::string outputDir;
    int numPartitions = 1;
    std::string method = "rcb";
    bool verbose = false;
    bool checkQuality = false;
    bool renumber = false;
    bool showBalance = false;
    Real scaleFactor = 1.0;
};

PreprocessOptions parseArguments(int argc, char* argv[]) {
    PreprocessOptions opts;
    
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
        } else if (arg == "-c" || arg == "--check") {
            opts.checkQuality = true;
        } else if (arg == "-r" || arg == "--renumber") {
            opts.renumber = true;
        } else if (arg == "-b" || arg == "--balance") {
            opts.showBalance = true;
        } else if (arg == "-p" || arg == "--partitions") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -p requires an argument\n";
                std::exit(1);
            }
            opts.numPartitions = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--method") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -m requires an argument\n";
                std::exit(1);
            }
            opts.method = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -o requires an argument\n";
                std::exit(1);
            }
            opts.outputDir = argv[++i];
        } else if (arg == "-s" || arg == "--scale") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -s requires an argument\n";
                std::exit(1);
            }
            opts.scaleFactor = std::stod(argv[++i]);
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
    
    if (opts.outputDir.empty()) {
        opts.outputDir = opts.caseDir;
    }
    
    return opts;
}

void checkMeshQuality(const Mesh& mesh, io::Logger& logger) {
    logger.info("Checking mesh quality...");
    
    // Compute quality metrics
    Real minVolume = std::numeric_limits<Real>::max();
    Real maxVolume = 0.0;
    Real totalVolume = 0.0;
    
    Real minSkewness = std::numeric_limits<Real>::max();
    Real maxSkewness = 0.0;
    Real avgSkewness = 0.0;
    
    Real minAspectRatio = std::numeric_limits<Real>::max();
    Real maxAspectRatio = 0.0;
    Real avgAspectRatio = 0.0;
    
    Index negativeCells = 0;
    
    for (Index i = 0; i < mesh.numCells(); ++i) {
        const Cell& cell = mesh.cell(i);
        Real volume = cell.volume();
        
        // Volume checks
        if (volume < 0) {
            negativeCells++;
            volume = -volume;
        }
        
        minVolume = std::min(minVolume, volume);
        maxVolume = std::max(maxVolume, volume);
        totalVolume += volume;
        
        // Compute skewness
        Real skewness = cell.skewness();
        minSkewness = std::min(minSkewness, skewness);
        maxSkewness = std::max(maxSkewness, skewness);
        avgSkewness += skewness;
        
        // Compute aspect ratio
        Real aspectRatio = cell.aspectRatio();
        minAspectRatio = std::min(minAspectRatio, aspectRatio);
        maxAspectRatio = std::max(maxAspectRatio, aspectRatio);
        avgAspectRatio += aspectRatio;
    }
    
    avgSkewness /= mesh.numCells();
    avgAspectRatio /= mesh.numCells();
    
    // Report results
    logger.info("Mesh Quality Report:");
    logger.info("  Cells: {}", mesh.numCells());
    logger.info("  Faces: {}", mesh.numFaces());
    logger.info("  Total Volume: {:.6e}", totalVolume);
    logger.info("  Volume Range: [{:.6e}, {:.6e}]", minVolume, maxVolume);
    
    if (negativeCells > 0) {
        logger.warning("  Negative Volume Cells: {} (CRITICAL!)", negativeCells);
    }
    
    logger.info("  Skewness: min={:.3f}, max={:.3f}, avg={:.3f}", 
                minSkewness, maxSkewness, avgSkewness);
    logger.info("  Aspect Ratio: min={:.3f}, max={:.3f}, avg={:.3f}",
                minAspectRatio, maxAspectRatio, avgAspectRatio);
    
    // Quality thresholds
    if (maxSkewness > 0.85) {
        logger.warning("  High skewness detected (>0.85) - may affect convergence");
    }
    if (maxAspectRatio > 100) {
        logger.warning("  High aspect ratio detected (>100) - may affect accuracy");
    }
}

void renumberCells(Mesh& mesh, io::Logger& logger) {
    logger.info("Renumbering cells for cache efficiency...");
    
    // Cuthill-McKee algorithm for bandwidth reduction
    std::vector<Index> newOrder(mesh.numCells());
    std::vector<bool> visited(mesh.numCells(), false);
    std::queue<Index> queue;
    
    // Find starting cell (minimum degree)
    Index startCell = 0;
    Index minDegree = mesh.numCells();
    
    for (Index i = 0; i < mesh.numCells(); ++i) {
        Index degree = mesh.cell(i).faces().size();
        if (degree < minDegree) {
            minDegree = degree;
            startCell = i;
        }
    }
    
    // BFS traversal
    Index newIndex = 0;
    queue.push(startCell);
    visited[startCell] = true;
    
    while (!queue.empty()) {
        Index current = queue.front();
        queue.pop();
        newOrder[newIndex++] = current;
        
        // Add unvisited neighbors
        std::vector<std::pair<Index, Index>> neighbors;
        
        for (const auto& face : mesh.cell(current).faces()) {
            if (!face.isBoundary()) {
                Index neighbor = (face.owner() == current) ? 
                               face.neighbor() : face.owner();
                if (!visited[neighbor]) {
                    Index degree = mesh.cell(neighbor).faces().size();
                    neighbors.emplace_back(degree, neighbor);
                }
            }
        }
        
        // Sort by degree
        std::sort(neighbors.begin(), neighbors.end());
        
        for (const auto& [degree, neighbor] : neighbors) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }
    
    // Handle any unvisited cells
    for (Index i = 0; i < mesh.numCells(); ++i) {
        if (!visited[i]) {
            newOrder[newIndex++] = i;
        }
    }
    
    // Apply renumbering
    mesh.renumberCells(newOrder);
    
    logger.info("Cell renumbering completed");
}

void writeMeshPartitions(const Mesh& mesh, const PreprocessOptions& opts,
                        io::Logger& logger) {
    logger.info("Writing mesh partitions...");
    
    // Create processor directories
    for (int proc = 0; proc < opts.numPartitions; ++proc) {
        std::string procDir = opts.outputDir + "/processor" + std::to_string(proc);
        
        // Create directory
        std::string command = "mkdir -p " + procDir;
        std::system(command.c_str());
        
        // Write mesh data for this processor
        std::ofstream meshFile(procDir + "/mesh.yaml");
        YAML::Emitter out;
        
        out << YAML::BeginMap;
        out << YAML::Key << "processor" << YAML::Value << proc;
        
        // Count cells for this processor
        Index numLocalCells = 0;
        std::vector<Index> localCells;
        
        for (Index i = 0; i < mesh.numCells(); ++i) {
            if (mesh.cell(i).processor() == proc) {
                numLocalCells++;
                localCells.push_back(i);
            }
        }
        
        out << YAML::Key << "numCells" << YAML::Value << numLocalCells;
        out << YAML::Key << "cells" << YAML::Value << YAML::Flow << localCells;
        
        // Write boundary patches
        out << YAML::Key << "boundaries" << YAML::Value << YAML::BeginSeq;
        for (const auto& patch : mesh.boundaryPatches()) {
            out << YAML::BeginMap;
            out << YAML::Key << "name" << YAML::Value << patch.name();
            out << YAML::Key << "type" << YAML::Value << patch.type();
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;
        
        out << YAML::EndMap;
        
        meshFile << out.c_str();
    }
    
    logger.info("Wrote {} processor directories", opts.numPartitions);
}

int main(int argc, char* argv[]) {
    // Initialize MPI if available
    parallel::MPIWrapper::initialize(argc, argv);
    
    try {
        // Parse command line arguments
        PreprocessOptions opts = parseArguments(argc, argv);
        
        // Setup logger
        auto logger = std::make_shared<io::Logger>("cfd-preprocess");
        if (opts.verbose) {
            logger->setLevel(io::Logger::Level::DEBUG);
        }
        
        logger->info("CFD Preprocessor");
        logger->info("Case directory: {}", opts.caseDir);
        
        // Read mesh
        logger->info("Reading mesh...");
        io::CaseReader reader(opts.caseDir);
        auto mesh = reader.readMesh();
        
        logger->info("Mesh statistics:");
        logger->info("  Cells: {}", mesh->numCells());
        logger->info("  Faces: {}", mesh->numFaces());
        logger->info("  Vertices: {}", mesh->numVertices());
        
        // Scale mesh if requested
        if (std::abs(opts.scaleFactor - 1.0) > 1e-10) {
            logger->info("Scaling mesh by factor {}", opts.scaleFactor);
            mesh->scale(opts.scaleFactor);
        }
        
        // Check mesh quality if requested
        if (opts.checkQuality) {
            checkMeshQuality(*mesh, *logger);
        }
        
        // Renumber cells if requested
        if (opts.renumber) {
            renumberCells(*mesh, *logger);
        }
        
        // Decompose mesh if multiple partitions requested
        if (opts.numPartitions > 1) {
            logger->info("Decomposing mesh into {} partitions using {} method",
                        opts.numPartitions, opts.method);
            
            parallel::DomainDecomposition decomposer;
            
            // Set decomposition method
            if (opts.method == "simple") {
                decomposer.setMethod(parallel::DomainDecomposition::Method::SIMPLE);
            } else if (opts.method == "rcb") {
                decomposer.setMethod(parallel::DomainDecomposition::Method::RCB);
            } else if (opts.method == "graph") {
                decomposer.setMethod(parallel::DomainDecomposition::Method::GRAPH);
            } else if (opts.method == "metis") {
                decomposer.setMethod(parallel::DomainDecomposition::Method::METIS);
            } else {
                logger->error("Unknown decomposition method: {}", opts.method);
                return 1;
            }
            
            // Perform decomposition
            decomposer.decompose(*mesh);
            
            // Show balance statistics if requested
            if (opts.showBalance) {
                logger->info("Load balance statistics:");
                logger->info("  Imbalance: {:.1f}%", 
                            decomposer.getLoadImbalance() * 100);
                logger->info("  Cut quality: {:.1f}%", 
                            decomposer.getCutQuality(*mesh) * 100);
            }
            
            // Write partitioned mesh
            writeMeshPartitions(*mesh, opts, *logger);
        }
        
        logger->info("Preprocessing completed successfully");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    parallel::MPIWrapper::finalize();
    return 0;
}
