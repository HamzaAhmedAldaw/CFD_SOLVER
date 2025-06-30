#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Mesh.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/physics/NavierStokes.hpp"
#include "cfd/solvers/NonlinearSolver.hpp"
#include "cfd/solvers/PressureVelocityCoupling.hpp"
#include "cfd/numerics/TimeIntegration.hpp"
#include "cfd/io/CaseReader.hpp"
#include "cfd/io/VTKWriter.hpp"
#include "cfd/io/Logger.hpp"

namespace cfd {

// Main CFD solver class
class CFDSolver {
public:
    struct SimulationSettings {
        // Time control
        Real startTime = 0.0;
        Real endTime = 1.0;
        Real deltaT = 0.001;
        bool adjustTimeStep = true;
        Real maxCo = 0.5;
        Real maxDeltaT = 1.0;
        
        // Solver control
        SolverType solverType = SolverType::INCOMPRESSIBLE;
        TimeScheme timeScheme = TimeScheme::BDF2;
        int maxIterations = 1000;
        Real convergenceTolerance = 1e-6;
        
        // Output control
        Real writeInterval = 0.1;
        int writeFrequency = 0;  // 0 = use interval
        std::string outputFormat = "VTK";
        bool writeResiduals = true;
        
        // Physics
        bool turbulence = false;
        TurbulenceModelType turbulenceModel = TurbulenceModelType::K_OMEGA_SST;
        Real referenceTemperature = 293.15;
        Real referencePressure = 101325.0;
    };
    
    CFDSolver(const std::string& caseDirectory);
    ~CFDSolver();
    
    // Setup and initialization
    void initialize();
    void readCase();
    void setupSolvers();
    
    // Main solution loop
    void run();
    void runSteady();
    void runTransient();
    
    // Access functions
    const Mesh& mesh() const { return *mesh_; }
    const SimulationSettings& settings() const { return settings_; }
    
private:
    // Case data
    std::string caseDir_;
    SimulationSettings settings_;
    SharedPtr<Mesh> mesh_;
    
    // Solution fields
    SharedPtr<VectorField> U_;      // Velocity
    SharedPtr<ScalarField> p_;      // Pressure
    SharedPtr<ScalarField> T_;      // Temperature
    SharedPtr<ScalarField> rho_;    // Density
    SharedPtr<ScalarField> mu_;     // Dynamic viscosity
    SharedPtr<ScalarField> k_;      // Turbulent kinetic energy
    SharedPtr<ScalarField> omega_;  // Specific dissipation rate
    
    // Derived fields
    SharedPtr<ScalarField> nut_;    // Turbulent viscosity
    SharedPtr<ScalarField> alphaEff_; // Effective thermal diffusivity
    
    // Physics models
    SharedPtr<physics::NavierStokes> nsEquations_;
    SharedPtr<physics::TurbulenceModel> turbulence_;
    SharedPtr<physics::Thermodynamics> thermo_;
    
    // Numerical schemes
    SharedPtr<numerics::TimeIntegration> timeIntegration_;
    SharedPtr<numerics::FluxScheme<CompressibleState>> fluxScheme_;
    SharedPtr<numerics::GradientScheme> gradScheme_;
    
    // Solvers
    SharedPtr<solvers::PressureVelocityCoupling> pvCoupling_;
    SharedPtr<solvers::NonlinearSolver> nonlinearSolver_;
    SharedPtr<solvers::LinearSolver> linearSolver_;
    
    // I/O
    SharedPtr<io::CaseReader> caseReader_;
    SharedPtr<io::VTKWriter> vtkWriter_;
    SharedPtr<io::Logger> logger_;
    
    // Time control
    Real currentTime_;
    Real deltaT_;
    int timeStep_;
    Real nextWriteTime_;
    
    // Convergence monitoring
    struct ResidualMonitor {
        std::map<std::string, Real> initial;
        std::map<std::string, Real> current;
        std::map<std::string, std::vector<Real>> history;
        
        void update(const std::string& field, Real residual);
        bool checkConvergence(Real tolerance) const;
        void print() const;
    } residuals_;
    
    // Helper functions
    void createFields();
    void setInitialConditions();
    void setBoundaryConditions();
    void updateProperties();
    void solveTimeStep();
    void updateTimeStep();
    bool writeNow() const;
    void writeFields();
    void writeResiduals();
    
    // Solver-specific routines
    void solveIncompressible();
    void solveCompressible();
    void solveLowMach();
    
    // Turbulence handling
    void solveTurbulence();
    void updateTurbulentViscosity();
};

// Main solver loop pseudocode implementation
/*
MAIN SOLVER LOOP PSEUDOCODE:

1. INITIALIZATION PHASE:
   - Read mesh and case settings
   - Create solution fields (U, p, T, etc.)
   - Set initial and boundary conditions
   - Initialize solvers and schemes
   - Setup parallel communication

2. TIME LOOP (for transient) or ITERATION LOOP (for steady):
   
   WHILE (time < endTime) OR (not converged):
      
      a. UPDATE TIME/ITERATION:
         - Increment time/iteration counter
         - Adjust time step based on CFL condition (if adaptive)
         - Store old time levels for temporal discretization
      
      b. UPDATE PROPERTIES:
         - Calculate density (equation of state)
         - Calculate viscosity (Sutherland/polynomial)
         - Update turbulent viscosity (if turbulent)
      
      c. SOLVE MOMENTUM-PRESSURE COUPLING:
         
         FOR each outer corrector (PIMPLE):
            
            i. MOMENTUM PREDICTOR:
               - Assemble momentum matrix with AD for Jacobian
               - Include convection, diffusion, pressure gradient
               - Solve for intermediate velocity U*
            
            ii. PRESSURE CORRECTION (PISO loop):
                FOR each pressure corrector:
                   - Apply Rhie-Chow interpolation
                   - Assemble pressure Laplacian
                   - Solve pressure correction equation
                   - Update pressure field
                   - Correct face fluxes
                   - Correct cell velocities
            
            iii. TURBULENCE EQUATIONS:
                 - Solve k-omega/k-epsilon transport
                 - Update turbulent viscosity
            
            iv. ENERGY EQUATION (if not isothermal):
                - Solve temperature transport
                - Update thermophysical properties
      
      d. CHECK CONVERGENCE:
         - Calculate residuals for all fields
         - Check against tolerance criteria
         - Update convergence history
      
      e. WRITE OUTPUT:
         - Check if write time reached
         - Output VTK/ParaView files
         - Write convergence history
         - Write restart data
      
      f. ADVANCE TIME:
         - Move to next time step
         - Update boundary conditions if time-dependent

3. FINALIZATION:
   - Write final fields
   - Output statistics
   - Clean up resources
*/

// Inline implementation of key solver loop
inline void CFDSolver::run() {
    logger_->info("Starting CFD simulation");
    logger_->info("Solver type: {}", 
                  settings_.solverType == SolverType::INCOMPRESSIBLE ? 
                  "Incompressible" : "Compressible");
    
    // Initialize simulation
    initialize();
    
    // Choose steady or transient
    if (settings_.timeScheme == TimeScheme::EULER_IMPLICIT && 
        settings_.endTime == settings_.startTime) {
        runSteady();
    } else {
        runTransient();
    }
    
    logger_->info("Simulation completed successfully");
}

inline void CFDSolver::runTransient() {
    logger_->info("Running transient simulation");
    logger_->info("Start time: {}, End time: {}", 
                  settings_.startTime, settings_.endTime);
    
    currentTime_ = settings_.startTime;
    timeStep_ = 0;
    
    // Time loop
    while (currentTime_ < settings_.endTime) {
        timeStep_++;
        
        // Adjust time step
        if (settings_.adjustTimeStep) {
            updateTimeStep();
        }
        
        // Ensure we don't overshoot end time
        deltaT_ = std::min(deltaT_, settings_.endTime - currentTime_);
        
        logger_->info("\nTime = {}, deltaT = {}, Step = {}", 
                      currentTime_ + deltaT_, deltaT_, timeStep_);
        
        // Store old time values
        U_->storeOldTime();
        p_->storeOldTime();
        if (T_) T_->storeOldTime();
        
        // Main solution step
        solveTimeStep();
        
        // Update time
        currentTime_ += deltaT_;
        
        // Write output if needed
        if (writeNow()) {
            writeFields();
            writeResiduals();
            nextWriteTime_ += settings_.writeInterval;
        }
        
        // Check for convergence (steady state)
        if (residuals_.checkConvergence(settings_.convergenceTolerance)) {
            logger_->info("Steady state reached at time = {}", currentTime_);
            break;
        }
    }
}

inline void CFDSolver::solveTimeStep() {
    // Update material properties
    updateProperties();
    
    // Select solver based on type
    switch (settings_.solverType) {
        case SolverType::INCOMPRESSIBLE:
            solveIncompressible();
            break;
        case SolverType::COMPRESSIBLE:
            solveCompressible();
            break;
        case SolverType::LOW_MACH:
            solveLowMach();
            break;
    }
    
    // Solve turbulence if enabled
    if (settings_.turbulence) {
        solveTurbulence();
    }
    
    // Update boundary conditions
    U_->updateBoundaryConditions();
    p_->updateBoundaryConditions();
}

} // namespace cfd
