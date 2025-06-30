#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Field.hpp"
#include <functional>
#include <vector>

namespace cfd::numerics {

// Base time integration class
class TimeIntegration {
public:
    TimeIntegration(TimeScheme scheme) : scheme_(scheme) {}
    virtual ~TimeIntegration() = default;
    
    // Advance solution in time
    virtual void advance(ScalarField& phi, 
                        const ScalarField& source,
                        Real dt) = 0;
    
    virtual void advance(VectorField& U, 
                        const VectorField& source,
                        Real dt) = 0;
    
    // Get time scheme
    TimeScheme scheme() const { return scheme_; }
    
    // Time derivative computation
    virtual Real timeDerivativeCoeff(int level = 0) const = 0;
    
protected:
    TimeScheme scheme_;
};

// Euler implicit (backward Euler)
class EulerImplicit : public TimeIntegration {
public:
    EulerImplicit() : TimeIntegration(TimeScheme::EULER_IMPLICIT) {}
    
    void advance(ScalarField& phi, const ScalarField& source, Real dt) override;
    void advance(VectorField& U, const VectorField& source, Real dt) override;
    
    Real timeDerivativeCoeff(int level = 0) const override {
        return (level == 0) ? 1.0 : -1.0;
    }
};

// Euler explicit (forward Euler)
class EulerExplicit : public TimeIntegration {
public:
    EulerExplicit() : TimeIntegration(TimeScheme::EULER_EXPLICIT) {}
    
    void advance(ScalarField& phi, const ScalarField& source, Real dt) override;
    void advance(VectorField& U, const VectorField& source, Real dt) override;
    
    Real timeDerivativeCoeff(int level = 0) const override {
        return 1.0;
    }
};

// Backward Differentiation Formula 2nd order (BDF2)
class BDF2 : public TimeIntegration {
public:
    BDF2() : TimeIntegration(TimeScheme::BDF2) {}
    
    void advance(ScalarField& phi, const ScalarField& source, Real dt) override;
    void advance(VectorField& U, const VectorField& source, Real dt) override;
    
    Real timeDerivativeCoeff(int level = 0) const override {
        // BDF2: (3φⁿ⁺¹ - 4φⁿ + φⁿ⁻¹)/(2Δt)
        switch (level) {
            case 0: return 1.5;    // φⁿ⁺¹
            case 1: return -2.0;   // φⁿ
            case 2: return 0.5;    // φⁿ⁻¹
            default: return 0.0;
        }
    }
    
    // Variable time step coefficients
    struct Coefficients {
        Real a0, a1, a2;  // Time derivative coefficients
        
        Coefficients(Real dtNew, Real dtOld) {
            Real r = dtNew / dtOld;
            a0 = (1.0 + 2.0*r) / (1.0 + r);
            a1 = -(1.0 + r);
            a2 = r*r / (1.0 + r);
        }
    };
};

// Backward Differentiation Formula 3rd order (BDF3)
class BDF3 : public TimeIntegration {
public:
    BDF3() : TimeIntegration(TimeScheme::BDF3) {}
    
    void advance(ScalarField& phi, const ScalarField& source, Real dt) override;
    void advance(VectorField& U, const VectorField& source, Real dt) override;
    
    Real timeDerivativeCoeff(int level = 0) const override {
        // BDF3: (11φⁿ⁺¹ - 18φⁿ + 9φⁿ⁻¹ - 2φⁿ⁻²)/(6Δt)
        switch (level) {
            case 0: return 11.0/6.0;   // φⁿ⁺¹
            case 1: return -18.0/6.0;  // φⁿ
            case 2: return 9.0/6.0;    // φⁿ⁻¹
            case 3: return -2.0/6.0;   // φⁿ⁻²
            default: return 0.0;
        }
    }
};

// Singly Diagonally Implicit Runge-Kutta (SDIRK) schemes
class SDIRK : public TimeIntegration {
public:
    struct ButcherTableau {
        MatrixX A;  // Runge-Kutta matrix
        VectorX b;  // Weights
        VectorX c;  // Nodes
        int stages;
        int order;
    };
    
    SDIRK(int order);
    
    void advance(ScalarField& phi, const ScalarField& source, Real dt) override;
    void advance(VectorField& U, const VectorField& source, Real dt) override;
    
    Real timeDerivativeCoeff(int level = 0) const override {
        return 1.0;  // For stage equations
    }
    
protected:
    ButcherTableau tableau_;
    
    // Stage residual function type
    using ResidualFunc = std::function<void(const ScalarField&, ScalarField&)>;
    
    // Solve stage equations
    void solveStage(ScalarField& phi, const ScalarField& phiOld,
                   const std::vector<ScalarField>& k,
                   int stage, Real dt, ResidualFunc residual);
    
private:
    void setupTableau(int order);
};

// 4th-order Runge-Kutta (RK4) - explicit
class RK4 : public TimeIntegration {
public:
    RK4() : TimeIntegration(TimeScheme::RK4) {}
    
    void advance(ScalarField& phi, const ScalarField& source, Real dt) override;
    void advance(VectorField& U, const VectorField& source, Real dt) override;
    
    Real timeDerivativeCoeff(int level = 0) const override {
        return 1.0;
    }
    
    // RK4 with custom RHS function
    template<typename FieldType>
    void advanceWithRHS(FieldType& phi, 
                       std::function<void(const FieldType&, FieldType&)> rhs,
                       Real dt);
};

// Strong Stability Preserving (SSP) Runge-Kutta schemes
class SSPRK : public TimeIntegration {
public:
    enum Order { SSP_RK2, SSP_RK3 };
    
    SSPRK(Order order) : TimeIntegration(TimeScheme::RK4), order_(order) {}
    
    void advance(ScalarField& phi, const ScalarField& source, Real dt) override;
    void advance(VectorField& U, const VectorField& source, Real dt) override;
    
    Real timeDerivativeCoeff(int level = 0) const override {
        return 1.0;
    }
    
private:
    Order order_;
};

// Dual time-stepping for unsteady simulations
class DualTimeStepping : public TimeIntegration {
public:
    DualTimeStepping(SharedPtr<TimeIntegration> physicalTime,
                     SharedPtr<TimeIntegration> pseudoTime)
        : TimeIntegration(physicalTime->scheme()),
          physicalTime_(physicalTime),
          pseudoTime_(pseudoTime) {}
    
    void advance(ScalarField& phi, const ScalarField& source, Real dt) override;
    void advance(VectorField& U, const VectorField& source, Real dt) override;
    
    Real timeDerivativeCoeff(int level = 0) const override {
        return physicalTime_->timeDerivativeCoeff(level);
    }
    
    // Pseudo-time stepping parameters
    void setPseudoTimeSteps(int steps) { pseudoTimeSteps_ = steps; }
    void setPseudoCFL(Real cfl) { pseudoCFL_ = cfl; }
    
private:
    SharedPtr<TimeIntegration> physicalTime_;
    SharedPtr<TimeIntegration> pseudoTime_;
    int pseudoTimeSteps_ = 50;
    Real pseudoCFL_ = 10.0;
    
    // Compute pseudo time step
    Real computePseudoTimeStep(const Field<Real>& phi) const;
};

// Local time stepping for steady-state acceleration
class LocalTimeStepping {
public:
    LocalTimeStepping(const Mesh& mesh, Real CFL = 1.0)
        : mesh_(mesh), CFL_(CFL) {}
    
    // Compute local time steps
    void computeTimeSteps(const VectorField& U,
                         const ScalarField& soundSpeed,
                         ScalarField& localDt) const;
    
    // Apply local time stepping
    template<typename FieldType>
    void advance(FieldType& phi, const FieldType& residual,
                const ScalarField& localDt);
    
private:
    const Mesh& mesh_;
    Real CFL_;
};

// Adaptive time stepping controller
class AdaptiveTimeStepController {
public:
    struct Parameters {
        Real targetCFL = 0.8;
        Real maxCFL = 1.0;
        Real safetyFactor = 0.9;
        Real maxIncrease = 1.2;
        Real maxDecrease = 0.5;
        Real minDt = 1e-10;
        Real maxDt = 1e10;
    };
    
    AdaptiveTimeStepController(const Parameters& params = Parameters())
        : params_(params) {}
    
    // Compute new time step
    Real computeTimeStep(Real currentDt, Real maxCourant) const;
    
    // Error-based time step control
    Real computeTimeStep(Real currentDt, Real errorEstimate,
                        Real tolerance, int order) const;
    
private:
    Parameters params_;
};

// Factory function
inline SharedPtr<TimeIntegration> createTimeIntegration(TimeScheme scheme) {
    switch (scheme) {
        case TimeScheme::EULER_IMPLICIT:
            return std::make_shared<EulerImplicit>();
        case TimeScheme::EULER_EXPLICIT:
            return std::make_shared<EulerExplicit>();
        case TimeScheme::BDF2:
            return std::make_shared<BDF2>();
        case TimeScheme::BDF3:
            return std::make_shared<BDF3>();
        case TimeScheme::SDIRK3:
            return std::make_shared<SDIRK>(3);
        case TimeScheme::SDIRK4:
            return std::make_shared<SDIRK>(4);
        case TimeScheme::RK4:
            return std::make_shared<RK4>();
        default:
            throw std::runtime_error("Unknown time integration scheme");
    }
}

} // namespace cfd::numerics