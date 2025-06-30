#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/core/Mesh.hpp"
#include "cfd/solvers/LinearSolver.hpp"
#include <memory>
#include <map>

namespace cfd::solvers {

// Base class for pressure-velocity coupling algorithms
class PressureVelocityCoupling {
public:
    struct Settings {
        // Solver control
        int maxIterations = 100;
        Real convergenceTolerance = 1e-5;
        
        // Under-relaxation factors
        Real velocityRelaxation = 0.7;
        Real pressureRelaxation = 0.3;
        
        // Algorithm specific
        bool momentumPredictor = true;
        int nCorrectors = 2;           // For PISO
        int nOuterCorrectors = 3;      // For PIMPLE
        bool turbulentPimple = false;  // For PIMPLE
        
        // Time stepping
        Real deltaT = 0.001;
        bool transient = true;
    };
    
    PressureVelocityCoupling(const Mesh& mesh);
    virtual ~PressureVelocityCoupling() = default;
    
    // Main solve function
    virtual void solve(VectorField& U, ScalarField& p,
                      const ScalarField& rho, const ScalarField& mu) = 0;
    
    // Get/set settings
    const Settings& settings() const { return settings_; }
    void setSettings(const Settings& settings) { settings_ = settings; }
    
    // Set linear solver
    void setLinearSolver(SharedPtr<LinearSolver> solver) {
        linearSolver_ = solver;
    }
    
protected:
    const Mesh& mesh_;
    Settings settings_;
    SharedPtr<LinearSolver> linearSolver_;
    
    // Working fields
    SharedPtr<ScalarField> phi_;      // Face flux
    SharedPtr<VectorField> HbyA_;     // H/A field
    SharedPtr<ScalarField> rAU_;      // 1/A_p field
    SharedPtr<VectorField> gradP_;    // Pressure gradient
    
    // Common operations
    void calculateMassFlux(const VectorField& U);
    void correctVelocity(VectorField& U, const ScalarField& p);
    void updateBoundaryConditions(VectorField& U, ScalarField& p);
    
    // Matrix assembly
    void assembleMomentumMatrix(const VectorField& U,
                               const ScalarField& rho,
                               const ScalarField& mu,
                               SparseMatrix& A,
                               VectorX& b);
    
    void assemblePressureMatrix(const ScalarField& rho,
                               SparseMatrix& A,
                               VectorX& b);
    
    // Rhie-Chow interpolation
    void rhieChowInterpolation(const VectorField& U,
                              const ScalarField& p,
                              ScalarField& phi);
};

// SIMPLE (Semi-Implicit Method for Pressure-Linked Equations)
class SIMPLESolver : public PressureVelocityCoupling {
public:
    using PressureVelocityCoupling::PressureVelocityCoupling;
    
    void solve(VectorField& U, ScalarField& p,
              const ScalarField& rho, const ScalarField& mu) override;
    
private:
    void momentumPredictor(VectorField& U, const ScalarField& p,
                          const ScalarField& rho, const ScalarField& mu);
    
    void pressureCorrector(const VectorField& U, ScalarField& p,
                          const ScalarField& rho);
    
    void velocityCorrector(VectorField& U, const ScalarField& p);
};

// PISO (Pressure Implicit with Splitting of Operators)
class PISOSolver : public PressureVelocityCoupling {
public:
    using PressureVelocityCoupling::PressureVelocityCoupling;
    
    void solve(VectorField& U, ScalarField& p,
              const ScalarField& rho, const ScalarField& mu) override;
    
private:
    void momentumPredictor(VectorField& U, const ScalarField& p,
                          const ScalarField& rho, const ScalarField& mu);
    
    void pressureCorrector(int corrector, VectorField& U, ScalarField& p,
                          const ScalarField& rho);
};

// PIMPLE (merged PISO-SIMPLE)
class PIMPLESolver : public PressureVelocityCoupling {
public:
    using PressureVelocityCoupling::PressureVelocityCoupling;
    
    void solve(VectorField& U, ScalarField& p,
              const ScalarField& rho, const ScalarField& mu) override;
    
private:
    // Residual control
    struct ResidualControl {
        Real initial = 0.0;
        Real current = 0.0;
        Real target = 1e-4;
        bool achieved = false;
    };
    
    std::map<std::string, ResidualControl> residuals_;
    
    void outerCorrectorLoop(VectorField& U, ScalarField& p,
                           const ScalarField& rho, const ScalarField& mu);
    
    bool checkResidualControls();
    void updateResiduals(const std::string& field, Real residual);
};

// Coupled solver (monolithic approach)
class CoupledSolver : public PressureVelocityCoupling {
public:
    using PressureVelocityCoupling::PressureVelocityCoupling;
    
    void solve(VectorField& U, ScalarField& p,
              const ScalarField& rho, const ScalarField& mu) override;
    
private:
    // Block matrix structure for coupled system
    void assembleBlockMatrix(const VectorField& U, const ScalarField& p,
                           const ScalarField& rho, const ScalarField& mu,
                           SparseMatrix& blockA, VectorX& blockB);
    
    // Extract solution components
    void extractSolution(const VectorX& blockX, VectorField& U, ScalarField& p);
};

// Fractional step method (projection method)
class FractionalStepSolver : public PressureVelocityCoupling {
public:
    enum ProjectionType { CHORIN, VAN_KAN, INCREMENTAL };
    
    FractionalStepSolver(const Mesh& mesh, ProjectionType type = VAN_KAN)
        : PressureVelocityCoupling(mesh), projectionType_(type) {}
    
    void solve(VectorField& U, ScalarField& p,
              const ScalarField& rho, const ScalarField& mu) override;
    
private:
    ProjectionType projectionType_;
    
    // Fractional step stages
    void predictorStep(VectorField& U, const ScalarField& p,
                      const ScalarField& rho, const ScalarField& mu);
    
    void projectionStep(const VectorField& U, ScalarField& p,
                       const ScalarField& rho);
    
    void correctionStep(VectorField& U, const ScalarField& p);
};

// Consistent solver for low-Mach number flows
class ConsistentSolver : public PressureVelocityCoupling {
public:
    using PressureVelocityCoupling::PressureVelocityCoupling;
    
    void solve(VectorField& U, ScalarField& p,
              const ScalarField& rho, const ScalarField& mu) override;
    
    // Additional fields for compressible effects
    void setThermodynamicPressure(Real p0) { p0_ = p0; }
    void setTemperatureField(SharedPtr<ScalarField> T) { T_ = T; }
    
private:
    Real p0_ = 101325.0;  // Thermodynamic pressure
    SharedPtr<ScalarField> T_;  // Temperature field
    
    // Density update based on equation of state
    void updateDensity(ScalarField& rho, const ScalarField& p);
};

// CFL-based adaptive time stepping
class CFLController {
public:
    CFLController(Real targetCourant = 0.8, Real maxDeltaT = 1.0)
        : targetCourant_(targetCourant), maxDeltaT_(maxDeltaT) {}
    
    // Compute time step based on CFL condition
    Real computeTimeStep(const VectorField& U, const Mesh& mesh,
                        Real currentDt) const;
    
    // Get maximum Courant number
    Real computeMaxCourant(const VectorField& U, const Mesh& mesh,
                          Real dt) const;
    
    // Set parameters
    void setTargetCourant(Real Co) { targetCourant_ = Co; }
    void setMaxDeltaT(Real dt) { maxDeltaT_ = dt; }
    
private:
    Real targetCourant_;
    Real maxDeltaT_;
    Real minDeltaT_ = 1e-10;
    Real growthFactor_ = 1.2;
};

// Factory function
SharedPtr<PressureVelocityCoupling> createPressureVelocitySolver(
    const std::string& type, const Mesh& mesh);

} // namespace cfd::solvers