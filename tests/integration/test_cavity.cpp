// tests/integration/test_cavity.cpp
#include <gtest/gtest.h>
#include "cfd/CFDSolver.hpp"
#include "cfd/io/CaseReader.hpp"
#include <filesystem>
#include <fstream>

using namespace cfd;

class CavityFlowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test case configuration
        caseConfig = R"({
            "simulation": {
                "type": "incompressible",
                "time": {
                    "start": 0.0,
                    "end": 0.1,
                    "dt": 0.001,
                    "writeInterval": 0.05
                }
            },
            "mesh": {
                "type": "structured",
                "dimensions": [10, 10],
                "domain": [[0, 0.1], [0, 0.1]]
            },
            "physics": {
                "model": "laminar",
                "properties": {
                    "density": 1.0,
                    "viscosity": 0.01
                }
            },
            "numerics": {
                "fluxScheme": "AUSM",
                "gradientScheme": "GreenGauss",
                "limiter": "Venkatakrishnan",
                "timeIntegration": "Euler"
            },
            "boundaryConditions": [
                {
                    "name": "top",
                    "type": "fixedValue",
                    "field": "velocity",
                    "value": [1.0, 0.0, 0.0]
                },
                {
                    "name": "walls",
                    "type": "noSlip"
                }
            ],
            "solvers": {
                "pressure": {
                    "type": "CG",
                    "tolerance": 1e-6,
                    "maxIterations": 100
                }
            }
        })";
        
        // Write config to file
        std::ofstream file("test_cavity.json");
        file << caseConfig;
        file.close();
    }
    
    void TearDown() override {
        // Clean up
        std::filesystem::remove("test_cavity.json");
        std::filesystem::remove_all("cavity_results");
    }

    std::string caseConfig;
};

TEST_F(CavityFlowTest, BasicRun) {
    // Read case
    CaseReader reader;
    auto caseSetup = reader.read("test_cavity.json");
    
    // Create and initialize solver
    CFDSolver solver(caseSetup);
    solver.initialize();
    
    // Run for a few time steps
    int steps = 0;
    while (solver.getCurrentTime() < 0.01 && steps < 10) {
        solver.step();
        steps++;
    }
    
    EXPECT_GT(steps, 0);
    EXPECT_LE(solver.getResidual(), 1.0); // Residual should be reasonable
}

TEST_F(CavityFlowTest, SteadyStateConvergence) {
    // Modify config for steady state
    auto config = caseConfig;
    // ... modify for steady state ...
    
    CaseReader reader;
    auto caseSetup = reader.read("test_cavity.json");
    
    CFDSolver solver(caseSetup);
    solver.initialize();
    
    // Run until convergence or max iterations
    int maxIter = 100;
    Real tolerance = 1e-4;
    
    for (int i = 0; i < maxIter; ++i) {
        solver.step();
        if (solver.getResidual() < tolerance) {
            break;
        }
    }
    
    EXPECT_LT(solver.getResidual(), tolerance * 10); // Should be close to converged
}
