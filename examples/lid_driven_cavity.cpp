// examples/lid_driven_cavity.cpp
// Complete example of setting up and running a lid-driven cavity simulation

#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include "cfd/CFDSolver.hpp"
#include "cfd/core/Mesh.hpp"
#include "cfd/io/CaseReader.hpp"
#include "cfd/io/VTKWriter.hpp"
#include "cfd/io/Logger.hpp"

using namespace cfd;

// Function to create a structured 2D mesh
std::shared_ptr<Mesh> createCavityMesh(int nx, int ny, Real L) {
    auto mesh = std::make_shared<Mesh>();
    
    // Create vertices
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            Real x = L * i / nx;
            Real y = L * j / ny;
            mesh->addVertex(Vector3d(x, y, 0.0));
        }
    }
    
    // Create cells (quadrilaterals)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = j * (nx + 1) + i;
            std::vector<Index> cell = {
                idx,           // bottom-left
                idx + 1,       // bottom-right
                idx + nx + 2,  // top-right
                idx + nx + 1   // top-left
            };
            mesh->addCell(cell);
        }
    }
    
    // Create boundary patches
    std::vector<Index> bottomWall, rightWall, topLid, leftWall;
    
    // Bottom wall
    for (int i = 0; i < nx; ++i) {
        bottomWall.push_back(i);
    }
    mesh->addBoundaryPatch("bottom", bottomWall);
    
    // Right wall
    for (int j = 0; j < ny; ++j) {
        rightWall.push_back(j * nx + (nx - 1));
    }
    mesh->addBoundaryPatch("right", rightWall);
    
    // Top lid (moving wall)
    for (int i = 0; i < nx; ++i) {
        topLid.push_back((ny - 1) * nx + i);
    }
    mesh->addBoundaryPatch("top", topLid);
    
    // Left wall
    for (int j = 0; j < ny; ++j) {
        leftWall.push_back(j * nx);
    }
    mesh->addBoundaryPatch("left", leftWall);
    
    mesh->finalize();
    return mesh;
}

// Function to create case configuration
void createCaseConfig(const std::string& filename, int nx, int ny, Real Re) {
    // Calculate viscosity from Reynolds number
    Real L = 1.0;      // Cavity length
    Real U = 1.0;      // Lid velocity
    Real rho = 1.0;    // Density
    Real nu = U * L / Re;  // Kinematic viscosity
    
    std::ofstream file(filename);
    file << R"({
    "simulation": {
        "name": "Lid-Driven Cavity Flow",
        "type": "incompressible",
        "steadyState": false,
        "time": {
            "start": 0.0,
            "end": 20.0,
            "dt": 0.005,
            "adjustTimeStep": true,
            "maxCo": 0.5,
            "writeInterval": 0.5
        }
    },
    
    "physics": {
        "model": "laminar",
        "properties": {
            "density": )" << rho << R"(,
            "viscosity": )" << nu << R"(
        }
    },
    
    "numerics": {
        "fluxScheme": "AUSM",
        "gradientScheme": "LeastSquares",
        "limiter": "Venkatakrishnan",
        "timeIntegration": "CrankNicolson",
        "pressureVelocityCoupling": "SIMPLE",
        "nCorrectors": 2,
        "nNonOrthogonalCorrectors": 1
    },
    
    "boundaryConditions": [
        {
            "name": "top",
            "type": "fixedValue",
            "field": "velocity",
            "value": [1.0, 0.0, 0.0]
        },
        {
            "name": "bottom",
            "type": "noSlip"
        },
        {
            "name": "left",
            "type": "noSlip"
        },
        {
            "name": "right",
            "type": "noSlip"
        },
        {
            "name": "all",
            "type": "zeroGradient",
            "field": "pressure"
        }
    ],
    
    "initialConditions": {
        "velocity": [0.0, 0.0, 0.0],
        "pressure": 0.0
    },
    
    "solvers": {
        "pressure": {
            "type": "PCG",
            "tolerance": 1e-6,
            "relTol": 0.01,
            "maxIterations": 1000,
            "preconditioner": "DIC"
        },
        "velocity": {
            "type": "BiCGSTAB",
            "tolerance": 1e-6,
            "relTol": 0.1,
            "maxIterations": 100,
            "preconditioner": "DILU"
        }
    },
    
    "output": {
        "format": "VTK",
        "fields": ["velocity", "pressure", "vorticity"],
        "probes": [
            {
                "name": "centerline_u",
                "type": "line",
                "start": [0.5, 0.0, 0.0],
                "end": [0.5, 1.0, 0.0],
                "field": "velocity",
                "component": 0,
                "nPoints": 100
            },
            {
                "name": "centerline_v",
                "type": "line", 
                "start": [0.0, 0.5, 0.0],
                "end": [1.0, 0.5, 0.0],
                "field": "velocity",
                "component": 1,
                "nPoints": 100
            }
        ]
    },
    
    "convergence": {
        "steadyTolerance": 1e-5,
        "checkInterval": 10,
        "minIterations": 100
    }
})";
    file.close();
}

// Custom progress monitor
class SimulationMonitor {
public:
    SimulationMonitor() : startTime(std::chrono::steady_clock::now()) {}
    
    void update(const CFDSolver& solver) {
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            currentTime - startTime).count();
        
        Real simTime = solver.getCurrentTime();
        Real residual = solver.getResidual();
        int iteration = solver.getIteration();
        
        // Clear line and print progress
        std::cout << "\r" << std::string(80, ' ') << "\r";
        std::cout << "Time: " << std::fixed << std::setprecision(3) << simTime 
                  << "s | Iter: " << iteration 
                  << " | Residual: " << std::scientific << std::setprecision(2) 
                  << residual
                  << " | Wall time: " << elapsed << "s" << std::flush;
                  
        // Log convergence history
        convergenceHistory.push_back({simTime, residual});
    }
    
    void saveConvergenceHistory(const std::string& filename) {
        std::ofstream file(filename);
        file << "# Time Residual\n";
        for (const auto& point : convergenceHistory) {
            file << point.first << " " << point.second << "\n";
        }
        file.close();
    }
    
private:
    std::chrono::steady_clock::time_point startTime;
    std::vector<std::pair<Real, Real>> convergenceHistory;
};

// Post-processing utilities
void extractCenterlineData(const CFDSolver& solver, const std::string& outputDir) {
    auto mesh = solver.getMesh();
    auto velocity = solver.getField("velocity");
    
    // Extract u-velocity along vertical centerline (x = 0.5)
    std::ofstream uFile(outputDir + "/centerline_u.dat");
    uFile << "# y u\n";
    
    for (int i = 0; i < 100; ++i) {
        Real y = i / 99.0;
        Vector3d point(0.5, y, 0.0);
        
        // Find closest cell
        Index cellId = mesh->findCell(point);
        if (cellId != InvalidIndex) {
            Vector3d vel = velocity->getValue(cellId);
            uFile << y << " " << vel[0] << "\n";
        }
    }
    uFile.close();
    
    // Extract v-velocity along horizontal centerline (y = 0.5)
    std::ofstream vFile(outputDir + "/centerline_v.dat");
    vFile << "# x v\n";
    
    for (int i = 0; i < 100; ++i) {
        Real x = i / 99.0;
        Vector3d point(x, 0.5, 0.0);
        
        Index cellId = mesh->findCell(point);
        if (cellId != InvalidIndex) {
            Vector3d vel = velocity->getValue(cellId);
            vFile << x << " " << vel[1] << "\n";
        }
    }
    vFile.close();
}

// Comparison with Ghia et al. benchmark data
void compareWithBenchmark(const std::string& outputDir, Real Re) {
    // Ghia et al. (1982) benchmark data for Re = 100, 400, 1000
    std::map<Real, std::vector<std::pair<Real, Real>>> ghiaData = {
        {100, {{0.0, 0.0}, {0.0625, -0.03717}, {0.1016, -0.04192}, 
               {0.5, 0.06050}, {0.9531, -0.5625}, {1.0, 1.0}}},
        {400, {{0.0, 0.0}, {0.0625, -0.08186}, {0.1016, -0.09266}, 
               {0.5, 0.10091}, {0.9531, -0.3869}, {1.0, 1.0}}},
        {1000, {{0.0, 0.0}, {0.0625, -0.10648}, {0.1016, -0.12317}, 
                {0.5, 0.11477}, {0.9531, -0.33304}, {1.0, 1.0}}}
    };
    
    if (ghiaData.find(Re) != ghiaData.end()) {
        std::ofstream compFile(outputDir + "/benchmark_comparison.dat");
        compFile << "# Comparison with Ghia et al. (1982) for Re = " << Re << "\n";
        compFile << "# y u_ghia u_computed error\n";
        
        // Load computed data and compare
        // ... comparison logic ...
        
        compFile.close();
    }
}

int main(int argc, char* argv[]) {
    try {
        // Initialize logger
        Logger& logger = Logger::getInstance();
        logger.setLevel(Logger::Level::INFO);
        
        // Simulation parameters
        int nx = 64;      // Grid resolution in x
        int ny = 64;      // Grid resolution in y
        Real Re = 1000;   // Reynolds number
        
        // Parse command line arguments
        if (argc > 1) nx = std::stoi(argv[1]);
        if (argc > 2) ny = std::stoi(argv[2]);
        if (argc > 3) Re = std::stod(argv[3]);
        
        logger.info("Starting lid-driven cavity simulation");
        logger.info("Grid: " + std::to_string(nx) + " x " + std::to_string(ny));
        logger.info("Reynolds number: " + std::to_string(Re));
        
        // Create output directory
        std::string outputDir = "cavity_Re" + std::to_string(int(Re));
        std::filesystem::create_directories(outputDir);
        
        // Create mesh
        logger.info("Creating mesh...");
        auto mesh = createCavityMesh(nx, ny, 1.0);
        logger.info("Mesh created with " + std::to_string(mesh->getNumCells()) + " cells");
        
        // Save mesh
        VTKWriter meshWriter;
        meshWriter.write(outputDir + "/mesh.vtk", *mesh);
        
        // Create case configuration
        std::string configFile = outputDir + "/case.json";
        createCaseConfig(configFile, nx, ny, Re);
        
        // Read case configuration
        CaseReader reader;
        auto caseSetup = reader.read(configFile);
        caseSetup->mesh = mesh;
        
        // Create and initialize solver
        logger.info("Initializing solver...");
        CFDSolver solver(caseSetup);
        solver.initialize();
        
        // Set up monitoring
        SimulationMonitor monitor;
        VTKWriter solutionWriter;
        
        // Time stepping loop
        logger.info("Starting time integration...");
        int writeIndex = 0;
        Real nextWriteTime = caseSetup->writeInterval;
        
        while (!solver.hasConverged() && solver.getCurrentTime() < caseSetup->endTime) {
            // Advance solution
            solver.step();
            
            // Update monitor
            monitor.update(solver);
            
            // Write solution at specified intervals
            if (solver.getCurrentTime() >= nextWriteTime) {
                std::string filename = outputDir + "/solution_" + 
                                     std::to_string(writeIndex++) + ".vtk";
                solutionWriter.write(filename, *mesh, solver.getFields());
                nextWriteTime += caseSetup->writeInterval;
                
                logger.info("\nSolution written at t = " + 
                           std::to_string(solver.getCurrentTime()));
            }
            
            // Check for early convergence (steady state)
            if (solver.getIteration() % 100 == 0 && solver.getResidual() < 1e-6) {
                logger.info("\nSteady state reached!");
                break;
            }
        }
        
        std::cout << "\n"; // New line after progress
        
        // Final solution
        logger.info("Writing final solution...");
        solutionWriter.write(outputDir + "/solution_final.vtk", 
                           *mesh, solver.getFields());
        
        // Post-processing
        logger.info("Post-processing...");
        extractCenterlineData(solver, outputDir);
        compareWithBenchmark(outputDir, Re);
        monitor.saveConvergenceHistory(outputDir + "/convergence.dat");
        
        // Summary
        logger.info("Simulation completed successfully!");
        logger.info("Results saved in: " + outputDir);
        
        // Performance metrics
        auto stats = solver.getStatistics();
        logger.info("Total iterations: " + std::to_string(stats.totalIterations));
        logger.info("Average CFL: " + std::to_string(stats.averageCFL));
        logger.info("Final residual: " + std::to_string(solver.getResidual()));
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
