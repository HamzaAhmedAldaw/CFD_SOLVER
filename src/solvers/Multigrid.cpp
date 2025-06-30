// === src/solvers/Multigrid.cpp ===
#include "cfd/solvers/Multigrid.hpp"
#include <algorithm>

namespace cfd::solvers {

MultigridSolver::MultigridSolver(const MultigridSettings& settings)
    : settings_(settings) {
    
    // Create smoothers for each level
    LinearSolver::SolverSettings smootherSettings;
    smootherSettings.maxIterations = settings.preSmoothingSteps;
    smootherSettings.tolerance = 1e-10; // Don't check convergence in smoother
    
    for (int level = 0; level < settings.numLevels; ++level) {
        smoothers_.push_back(
            LinearSolver::create(LinearSolverType::BICGSTAB, smootherSettings));
    }
}

void MultigridSolver::setup(const std::vector<SparseMatrix>& A_levels) {
    A_levels_ = A_levels;
    numLevels_ = A_levels.size();
    
    // Setup prolongation and restriction operators
    setupTransferOperators();
    
    // Allocate work vectors
    r_levels_.resize(numLevels_);
    e_levels_.resize(numLevels_);
    
    for (int level = 0; level < numLevels_; ++level) {
        Index size = A_levels_[level].rows();
        r_levels_[level] = VectorX::Zero(size);
        e_levels_[level] = VectorX::Zero(size);
    }
}

bool MultigridSolver::solve(const SparseMatrix& A, const VectorX& b, VectorX& x) {
    // V-cycle or W-cycle
    iterations_ = 0;
    
    // Initial residual
    VectorX r = b - A * x;
    Real residual0 = r.norm();
    
    while (iterations_ < settings_.maxIterations) {
        iterations_++;
        
        // Perform one multigrid cycle
        if (settings_.cycleType == CycleType::V_CYCLE) {
            vCycle(0, x, b);
        } else {
            wCycle(0, x, b);
        }
        
        // Check convergence
        r = b - A * x;
        Real residual = r.norm();
        
        if (residual < settings_.tolerance ||
            residual / residual0 < settings_.relativeTolerance) {
            finalResidual_ = residual;
            return true;
        }
    }
    
    finalResidual_ = r.norm();
    return false;
}

void MultigridSolver::vCycle(int level, VectorX& x, const VectorX& b) {
    if (level == numLevels_ - 1) {
        // Coarsest level - solve exactly
        smoothers_[level]->solve(A_levels_[level], b, x);
        return;
    }
    
    // Pre-smoothing
    for (int i = 0; i < settings_.preSmoothingSteps; ++i) {
        smoothers_[level]->solve(A_levels_[level], b, x);
    }
    
    // Compute residual
    r_levels_[level] = b - A_levels_[level] * x;
    
    // Restrict residual to coarse grid
    r_levels_[level + 1] = restriction(level, r_levels_[level]);
    
    // Solve coarse grid problem
    e_levels_[level + 1].setZero();
    vCycle(level + 1, e_levels_[level + 1], r_levels_[level + 1]);
    
    // Prolongate correction to fine grid
    VectorX correction = prolongation(level, e_levels_[level + 1]);
    
    // Apply correction
    x += correction;
    
    // Post-smoothing
    for (int i = 0; i < settings_.postSmoothingSteps; ++i) {
        smoothers_[level]->solve(A_levels_[level], b, x);
    }
}

void MultigridSolver::wCycle(int level, VectorX& x, const VectorX& b) {
    if (level == numLevels_ - 1) {
        // Coarsest level
        smoothers_[level]->solve(A_levels_[level], b, x);
        return;
    }
    
    // Pre-smoothing
    for (int i = 0; i < settings_.preSmoothingSteps; ++i) {
        smoothers_[level]->solve(A_levels_[level], b, x);
    }
    
    // Compute residual
    r_levels_[level] = b - A_levels_[level] * x;
    
    // Restrict
    r_levels_[level + 1] = restriction(level, r_levels_[level]);
    
    // Two coarse grid corrections (W-cycle)
    e_levels_[level + 1].setZero();
    wCycle(level + 1, e_levels_[level + 1], r_levels_[level + 1]);
    wCycle(level + 1, e_levels_[level + 1], r_levels_[level + 1]);
    
    // Prolongate and correct
    x += prolongation(level, e_levels_[level + 1]);
    
    // Post-smoothing
    for (int i = 0; i < settings_.postSmoothingSteps; ++i) {
        smoothers_[level]->solve(A_levels_[level], b, x);
    }
}

void MultigridSolver::setupTransferOperators() {
    // Setup prolongation and restriction operators
    // This is a simplified implementation
    // Full implementation would use mesh hierarchy
    
    P_levels_.clear();
    R_levels_.clear();
    
    for (int level = 0; level < numLevels_ - 1; ++level) {
        Index nFine = A_levels_[level].rows();
        Index nCoarse = A_levels_[level + 1].rows();
        
        // Simple injection for now
        SparseMatrix P(nFine, nCoarse);
        SparseMatrix R(nCoarse, nFine);
        
        // Build operators (simplified)
        std::vector<Triplet> p_triplets, r_triplets;
        
        Real ratio = Real(nFine) / Real(nCoarse);
        for (Index i = 0; i < nCoarse; ++i) {
            Index j = i * ratio;
            if (j < nFine) {
                p_triplets.emplace_back(j, i, 1.0);
                r_triplets.emplace_back(i, j, 1.0);
            }
        }
        
        P.setFromTriplets(p_triplets.begin(), p_triplets.end());
        R.setFromTriplets(r_triplets.begin(), r_triplets.end());
        
        P_levels_.push_back(P);
        R_levels_.push_back(R);
    }
}

VectorX MultigridSolver::restriction(int level, const VectorX& fine) {
    return R_levels_[level] * fine;
}

VectorX MultigridSolver::prolongation(int level, const VectorX& coarse) {
    return P_levels_[level] * coarse;
}

// Algebraic Multigrid
AlgebraicMultigrid::AlgebraicMultigrid(const MultigridSettings& settings)
    : MultigridSolver(settings) {
}

void AlgebraicMultigrid::setup(const SparseMatrix& A) {
    A_levels_.clear();
    A_levels_.push_back(A);
    
    // Coarsen matrix algebraically
    for (int level = 0; level < settings_.numLevels - 1; ++level) {
        SparseMatrix A_coarse = coarsenMatrix(A_levels_[level]);
        A_levels_.push_back(A_coarse);
        
        if (A_coarse.rows() < settings_.coarsestSize) {
            break;
        }
    }
    
    numLevels_ = A_levels_.size();
    MultigridSolver::setup(A_levels_);
}

SparseMatrix AlgebraicMultigrid::coarsenMatrix(const SparseMatrix& A_fine) {
    // Classical AMG coarsening
    // Simplified implementation - full AMG is complex
    
    const Index n = A_fine.rows();
    
    // Compute strength of connection
    std::vector<std::vector<Index>> strongConnections(n);
    
    for (Index i = 0; i < n; ++i) {
        Real maxOffDiag = 0.0;
        
        for (SparseMatrix::InnerIterator it(A_fine, i); it; ++it) {
            if (it.row() != i) {
                maxOffDiag = std::max(maxOffDiag, std::abs(it.value()));
            }
        }
        
        Real threshold = settings_.strongThreshold * maxOffDiag;
        
        for (SparseMatrix::InnerIterator it(A_fine, i); it; ++it) {
            if (it.row() != i && std::abs(it.value()) >= threshold) {
                strongConnections[i].push_back(it.row());
            }
        }
    }
    
    // Select coarse points (simplified)
    std::vector<bool> isCoarse(n, false);
    Index nCoarse = 0;
    
    for (Index i = 0; i < n; i += 2) { // Simple coarsening
        isCoarse[i] = true;
        nCoarse++;
    }
    
    // Build coarse matrix (Galerkin projection: A_c = P^T * A_f * P)
    // Simplified for demonstration
    SparseMatrix A_coarse(nCoarse, nCoarse);
    
    // ... (full implementation would be more complex)
    
    return A_coarse;
}

} // namespace cfd::solvers
