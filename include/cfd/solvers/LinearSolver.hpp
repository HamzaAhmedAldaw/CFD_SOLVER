#pragma once

#include "cfd/core/Types.hpp"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <memory>

namespace cfd::solvers {

// Solver result structure
struct LinearSolverResult {
    bool converged;
    int iterations;
    Real residual;
    Real relativeResidual;
    Real time;  // Solution time in seconds
};

// Base linear solver class
class LinearSolver {
public:
    struct Settings {
        LinearSolverType type = LinearSolverType::GMRES;
        PreconditionerType preconditioner = PreconditionerType::ILU0;
        int maxIterations = 1000;
        Real tolerance = 1e-6;
        Real relativeTolerance = 1e-6;
        int restartSize = 30;  // For GMRES
        int fillLevel = 0;     // For ILU(k)
        bool verbose = false;
    };
    
    LinearSolver(const Settings& settings = Settings())
        : settings_(settings) {}
    
    virtual ~LinearSolver() = default;
    
    // Solve Ax = b
    virtual LinearSolverResult solve(const SparseMatrix& A,
                                   const VectorX& b,
                                   VectorX& x) = 0;
    
    // Get settings
    const Settings& settings() const { return settings_; }
    void setSettings(const Settings& settings) { settings_ = settings; }
    
protected:
    Settings settings_;
    
    // Check convergence
    bool checkConvergence(Real residual, Real residual0, int iter) const;
};

// GMRES (Generalized Minimal Residual) solver
class GMRESSolver : public LinearSolver {
public:
    GMRESSolver(const Settings& settings = Settings())
        : LinearSolver(settings) {
        settings_.type = LinearSolverType::GMRES;
    }
    
    LinearSolverResult solve(const SparseMatrix& A,
                           const VectorX& b,
                           VectorX& x) override;
    
private:
    // GMRES implementation with restart
    void gmresRestarted(const SparseMatrix& A,
                       const VectorX& b,
                       VectorX& x,
                       int& iterations,
                       Real& residual);
    
    // Arnoldi process
    void arnoldiProcess(const SparseMatrix& A,
                       const VectorX& r0,
                       MatrixX& H,
                       MatrixX& V,
                       int k);
    
    // Apply preconditioner
    VectorX applyPreconditioner(const VectorX& r) const;
};

// BiCGSTAB (Bi-Conjugate Gradient Stabilized) solver
class BiCGSTABSolver : public LinearSolver {
public:
    BiCGSTABSolver(const Settings& settings = Settings())
        : LinearSolver(settings) {
        settings_.type = LinearSolverType::BICGSTAB;
    }
    
    LinearSolverResult solve(const SparseMatrix& A,
                           const VectorX& b,
                           VectorX& x) override;
};

// Conjugate Gradient solver (for symmetric positive definite matrices)
class CGSolver : public LinearSolver {
public:
    CGSolver(const Settings& settings = Settings())
        : LinearSolver(settings) {
        settings_.type = LinearSolverType::CG;
    }
    
    LinearSolverResult solve(const SparseMatrix& A,
                           const VectorX& b,
                           VectorX& x) override;
};

// Direct solver using LU decomposition
class DirectSolver : public LinearSolver {
public:
    DirectSolver(const Settings& settings = Settings())
        : LinearSolver(settings) {
        settings_.type = LinearSolverType::DIRECT;
    }
    
    LinearSolverResult solve(const SparseMatrix& A,
                           const VectorX& b,
                           VectorX& x) override;
    
private:
    mutable Eigen::SparseLU<SparseMatrix> lu_;
    mutable bool factorized_ = false;
};

// Preconditioner base class
class Preconditioner {
public:
    virtual ~Preconditioner() = default;
    
    // Apply preconditioner: solve Mz = r
    virtual VectorX apply(const VectorX& r) const = 0;
    
    // Setup preconditioner for matrix A
    virtual void setup(const SparseMatrix& A) = 0;
};

// Jacobi (diagonal) preconditioner
class JacobiPreconditioner : public Preconditioner {
public:
    VectorX apply(const VectorX& r) const override {
        return r.cwiseProduct(invDiag_);
    }
    
    void setup(const SparseMatrix& A) override;
    
private:
    VectorX invDiag_;
};

// ILU(0) preconditioner
class ILU0Preconditioner : public Preconditioner {
public:
    VectorX apply(const VectorX& r) const override;
    void setup(const SparseMatrix& A) override;
    
private:
    SparseMatrix L_, U_;
    VectorX invDiag_;
};

// ILU(k) preconditioner with fill level k
class ILUKPreconditioner : public Preconditioner {
public:
    ILUKPreconditioner(int fillLevel = 1) : fillLevel_(fillLevel) {}
    
    VectorX apply(const VectorX& r) const override;
    void setup(const SparseMatrix& A) override;
    
private:
    int fillLevel_;
    SparseMatrix LU_;
    std::vector<int> levelOfFill_;
};

// Gauss-Seidel preconditioner
class GaussSeidelPreconditioner : public Preconditioner {
public:
    GaussSeidelPreconditioner(int sweeps = 1) : sweeps_(sweeps) {}
    
    VectorX apply(const VectorX& r) const override;
    void setup(const SparseMatrix& A) override;
    
private:
    int sweeps_;
    SharedPtr<SparseMatrix> A_;
};

// AMG (Algebraic Multigrid) preconditioner
class AMGPreconditioner : public Preconditioner {
public:
    struct Settings {
        int maxLevels = 10;
        Real coarseningRatio = 0.25;
        int preSmoothingSteps = 2;
        int postSmoothingSteps = 2;
        int coarseSolverIterations = 20;
        Real strongThreshold = 0.25;
    };
    
    AMGPreconditioner(const Settings& settings = Settings())
        : settings_(settings) {}
    
    VectorX apply(const VectorX& r) const override;
    void setup(const SparseMatrix& A) override;
    
private:
    Settings settings_;
    
    struct Level {
        SparseMatrix A;          // System matrix
        SparseMatrix P;          // Prolongation operator
        SparseMatrix R;          // Restriction operator
        SharedPtr<Preconditioner> smoother;  // Smoother
    };
    
    std::vector<Level> levels_;
    
    // AMG setup phases
    void generateHierarchy(const SparseMatrix& A);
    SparseMatrix coarsen(const SparseMatrix& A, SparseMatrix& P);
    void computeStrongConnections(const SparseMatrix& A,
                                 std::vector<std::set<int>>& strong);
    
    // V-cycle
    VectorX vCycle(int level, const VectorX& r) const;
};

// Block preconditioners for coupled systems
template<int BlockSize>
class BlockPreconditioner : public Preconditioner {
public:
    VectorX apply(const VectorX& r) const override;
    void setup(const SparseMatrix& A) override;
    
private:
    std::vector<Matrix<Real, BlockSize, BlockSize>> blocks_;
    std::vector<Matrix<Real, BlockSize, BlockSize>> invBlocks_;
};

// Factory function for preconditioners
inline SharedPtr<Preconditioner> createPreconditioner(
    PreconditionerType type,
    const LinearSolver::Settings& settings = LinearSolver::Settings()) {
    
    switch (type) {
        case PreconditionerType::NONE:
            return nullptr;
        case PreconditionerType::JACOBI:
            return std::make_shared<JacobiPreconditioner>();
        case PreconditionerType::ILU0:
            return std::make_shared<ILU0Preconditioner>();
        case PreconditionerType::ILUK:
            return std::make_shared<ILUKPreconditioner>(settings.fillLevel);
        case PreconditionerType::GAUSS_SEIDEL:
            return std::make_shared<GaussSeidelPreconditioner>();
        case PreconditionerType::AMG:
            return std::make_shared<AMGPreconditioner>();
        default:
            throw std::runtime_error("Unknown preconditioner type");
    }
}

// Factory function for linear solvers
inline SharedPtr<LinearSolver> createLinearSolver(
    const LinearSolver::Settings& settings = LinearSolver::Settings()) {
    
    switch (settings.type) {
        case LinearSolverType::GMRES:
            return std::make_shared<GMRESSolver>(settings);
        case LinearSolverType::BICGSTAB:
            return std::make_shared<BiCGSTABSolver>(settings);
        case LinearSolverType::CG:
            return std::make_shared<CGSolver>(settings);
        case LinearSolverType::DIRECT:
            return std::make_shared<DirectSolver>(settings);
        default:
            throw std::runtime_error("Unknown linear solver type");
    }
}

} // namespace cfd::solvers