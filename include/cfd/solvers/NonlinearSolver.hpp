#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/solvers/LinearSolver.hpp"
#include "cfd/ad/DualNumber.hpp"
#include <functional>
#include <vector>

namespace cfd::solvers {

// Nonlinear solver result
struct NonlinearSolverResult {
    bool converged;
    int iterations;
    Real residual;
    Real relativeResidual;
    std::vector<Real> residualHistory;
    Real time;  // Solution time in seconds
};

// Base nonlinear solver class
class NonlinearSolver {
public:
    struct Settings {
        int maxIterations = 50;
        Real tolerance = 1e-8;
        Real relativeTolerance = 1e-8;
        Real relaxationFactor = 1.0;
        bool useLineSearch = false;
        bool verbose = true;
        
        // Linear solver settings
        LinearSolver::Settings linearSettings;
    };
    
    NonlinearSolver(const Settings& settings = Settings())
        : settings_(settings) {
        linearSolver_ = createLinearSolver(settings_.linearSettings);
    }
    
    virtual ~NonlinearSolver() = default;
    
    // Solve F(x) = 0
    virtual NonlinearSolverResult solve(
        VectorX& x,
        std::function<void(const VectorX&, VectorX&)> residualFunc,
        std::function<void(const VectorX&, SparseMatrix&)> jacobianFunc) = 0;
    
    // Get/set settings
    const Settings& settings() const { return settings_; }
    void setSettings(const Settings& settings) { settings_ = settings; }
    
protected:
    Settings settings_;
    SharedPtr<LinearSolver> linearSolver_;
    
    // Helper functions
    bool checkConvergence(Real residual, Real residual0, int iter) const;
    Real computeNorm(const VectorX& v) const;
};

// Newton-Raphson solver
class NewtonSolver : public NonlinearSolver {
public:
    using NonlinearSolver::NonlinearSolver;
    
    NonlinearSolverResult solve(
        VectorX& x,
        std::function<void(const VectorX&, VectorX&)> residualFunc,
        std::function<void(const VectorX&, SparseMatrix&)> jacobianFunc) override;
    
private:
    // Line search for globalization
    Real lineSearch(const VectorX& x, const VectorX& dx,
                   std::function<void(const VectorX&, VectorX&)> residualFunc,
                   Real residualNorm);
};

// Inexact Newton solver (with approximated Jacobian)
class InexactNewtonSolver : public NewtonSolver {
public:
    InexactNewtonSolver(const Settings& settings = Settings())
        : NewtonSolver(settings) {
        forcingSequence_ = {0.1, 0.01, 0.001};  // Eisenstat-Walker
    }
    
    NonlinearSolverResult solve(
        VectorX& x,
        std::function<void(const VectorX&, VectorX&)> residualFunc,
        std::function<void(const VectorX&, SparseMatrix&)> jacobianFunc) override;
    
    // Set forcing sequence for inexact linear solves
    void setForcingSequence(const std::vector<Real>& sequence) {
        forcingSequence_ = sequence;
    }
    
private:
    std::vector<Real> forcingSequence_;
    
    // Compute forcing term
    Real computeForcingTerm(int iter, Real residualRatio) const;
};

// Quasi-Newton solver (BFGS/L-BFGS)
class QuasiNewtonSolver : public NonlinearSolver {
public:
    enum Method { BFGS, LBFGS, BROYDEN };
    
    QuasiNewtonSolver(Method method = LBFGS, int memorySize = 10,
                      const Settings& settings = Settings())
        : NonlinearSolver(settings), method_(method), memorySize_(memorySize) {}
    
    NonlinearSolverResult solve(
        VectorX& x,
        std::function<void(const VectorX&, VectorX&)> residualFunc,
        std::function<void(const VectorX&, SparseMatrix&)> jacobianFunc) override;
    
private:
    Method method_;
    int memorySize_;  // For L-BFGS
    
    // BFGS update
    void updateBFGS(MatrixX& H, const VectorX& s, const VectorX& y);
    
    // L-BFGS two-loop recursion
    VectorX lbfgsTwoLoop(const VectorX& grad,
                        const std::deque<VectorX>& s_history,
                        const std::deque<VectorX>& y_history);
    
    // Broyden update
    void updateBroyden(SparseMatrix& J, const VectorX& s, const VectorX& y);
};

// Picard iteration (fixed-point iteration)
class PicardSolver : public NonlinearSolver {
public:
    using NonlinearSolver::NonlinearSolver;
    
    NonlinearSolverResult solve(
        VectorX& x,
        std::function<void(const VectorX&, VectorX&)> residualFunc,
        std::function<void(const VectorX&, SparseMatrix&)> jacobianFunc) override;
    
    // Solve x = G(x) form
    NonlinearSolverResult solveFixedPoint(
        VectorX& x,
        std::function<void(const VectorX&, VectorX&)> updateFunc);
};

// Anderson acceleration for fixed-point iterations
class AndersonAcceleratedSolver : public NonlinearSolver {
public:
    AndersonAcceleratedSolver(int depth = 5, const Settings& settings = Settings())
        : NonlinearSolver(settings), depth_(depth) {}
    
    NonlinearSolverResult solve(
        VectorX& x,
        std::function<void(const VectorX&, VectorX&)> residualFunc,
        std::function<void(const VectorX&, SparseMatrix&)> jacobianFunc) override;
    
private:
    int depth_;  // Anderson depth
    
    // Anderson mixing
    VectorX andersonMixing(const std::deque<VectorX>& F_history,
                          const std::deque<VectorX>& x_history);
};

// Trust region solver
class TrustRegionSolver : public NonlinearSolver {
public:
    TrustRegionSolver(Real initialRadius = 1.0, const Settings& settings = Settings())
        : NonlinearSolver(settings), trustRadius_(initialRadius) {}
    
    NonlinearSolverResult solve(
        VectorX& x,
        std::function<void(const VectorX&, VectorX&)> residualFunc,
        std::function<void(const VectorX&, SparseMatrix&)> jacobianFunc) override;
    
private:
    Real trustRadius_;
    
    // Trust region subproblem solver
    VectorX solveTrustRegionSubproblem(const SparseMatrix& J,
                                       const VectorX& g,
                                       Real radius);
    
    // Update trust region radius
    void updateTrustRadius(Real actualReduction, Real predictedReduction,
                          Real& radius, bool& accepted);
};

// Nonlinear solver for field equations
template<typename T>
class FieldNonlinearSolver {
public:
    using ResidualFunc = std::function<void(const Field<T>&, Field<T>&)>;
    using JacobianFunc = std::function<void(const Field<T>&, SparseMatrix&)>;
    
    FieldNonlinearSolver(const Mesh& mesh, 
                        SharedPtr<NonlinearSolver> solver)
        : mesh_(mesh), solver_(solver) {}
    
    // Solve field equation
    NonlinearSolverResult solve(Field<T>& phi,
                               ResidualFunc residualFunc,
                               JacobianFunc jacobianFunc);
    
private:
    const Mesh& mesh_;
    SharedPtr<NonlinearSolver> solver_;
    
    // Convert between field and vector representations
    void fieldToVector(const Field<T>& field, VectorX& vec) const;
    void vectorToField(const VectorX& vec, Field<T>& field) const;
};

// Jacobian computation using automatic differentiation
template<typename T>
class ADJacobian {
public:
    using DualField = Field<ad::DualNumber<T>>;
    
    ADJacobian(const Mesh& mesh) : mesh_(mesh) {}
    
    // Compute Jacobian using forward-mode AD
    void computeJacobian(
        const Field<T>& phi,
        std::function<void(const DualField&, DualField&)> residualFunc,
        SparseMatrix& jacobian);
    
private:
    const Mesh& mesh_;
    
    // Seed dual numbers for differentiation
    void seedDualNumbers(DualField& dualPhi, Index variable);
    
    // Extract derivatives
    void extractDerivatives(const DualField& residual,
                           Index variable,
                           std::vector<Triplet>& triplets);
};

// Jacobian-free Newton-Krylov (JFNK) solver
class JFNKSolver : public NonlinearSolver {
public:
    JFNKSolver(const Settings& settings = Settings())
        : NonlinearSolver(settings), epsilon_(1e-8) {}
    
    NonlinearSolverResult solve(
        VectorX& x,
        std::function<void(const VectorX&, VectorX&)> residualFunc,
        std::function<void(const VectorX&, SparseMatrix&)> jacobianFunc) override;
    
    // Set finite difference epsilon
    void setEpsilon(Real eps) { epsilon_ = eps; }
    
private:
    Real epsilon_;
    
    // Matrix-free Jacobian-vector product
    class JacobianVectorProduct {
    public:
        JacobianVectorProduct(
            const VectorX& x,
            std::function<void(const VectorX&, VectorX&)> residualFunc,
            Real epsilon)
            : x_(x), residualFunc_(residualFunc), epsilon_(epsilon) {}
        
        // Compute J*v using finite differences
        VectorX apply(const VectorX& v) const;
        
    private:
        const VectorX& x_;
        std::function<void(const VectorX&, VectorX&)> residualFunc_;
        Real epsilon_;
    };
};

// Factory function
inline SharedPtr<NonlinearSolver> createNonlinearSolver(
    const std::string& type,
    const NonlinearSolver::Settings& settings = NonlinearSolver::Settings()) {
    
    if (type == "Newton") {
        return std::make_shared<NewtonSolver>(settings);
    } else if (type == "InexactNewton") {
        return std::make_shared<InexactNewtonSolver>(settings);
    } else if (type == "QuasiNewton") {
        return std::make_shared<QuasiNewtonSolver>(QuasiNewtonSolver::LBFGS, 10, settings);
    } else if (type == "Picard") {
        return std::make_shared<PicardSolver>(settings);
    } else if (type == "Anderson") {
        return std::make_shared<AndersonAcceleratedSolver>(5, settings);
    } else if (type == "TrustRegion") {
        return std::make_shared<TrustRegionSolver>(1.0, settings);
    } else if (type == "JFNK") {
        return std::make_shared<JFNKSolver>(settings);
    } else {
        throw std::runtime_error("Unknown nonlinear solver type: " + type);
    }
}

} // namespace cfd::solvers