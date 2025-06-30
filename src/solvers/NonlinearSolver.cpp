// === src/solvers/NonlinearSolver.cpp ===
#include "cfd/solvers/NonlinearSolver.hpp"
#include "cfd/ad/AutoDiff.hpp"
#include <algorithm>

namespace cfd::solvers {

NonlinearSolver::NonlinearSolver(SharedPtr<Mesh> mesh,
                               SharedPtr<LinearSolver> linearSolver)
    : mesh_(mesh), linearSolver_(linearSolver),
      maxIterations_(50), tolerance_(1e-6), verbose_(false) {
}

bool NonlinearSolver::solve(std::vector<ScalarField>& fields,
                           ResidualFunction residualFunc) {
    const Index numFields = fields.size();
    const Index numCells = mesh_->numCells();
    const Index totalDOF = numFields * numCells;
    
    // Pack fields into solution vector
    VectorX x(totalDOF);
    packFields(fields, x);
    
    // Newton-Raphson iteration
    for (int iter = 0; iter < maxIterations_; ++iter) {
        // Compute residual
        std::vector<ScalarField> residualFields = residualFunc(fields);
        
        VectorX F(totalDOF);
        packFields(residualFields, F);
        
        // Check convergence
        Real residualNorm = F.norm();
        if (verbose_) {
            logger_->info("Nonlinear iteration {}: residual = {:.3e}", 
                         iter, residualNorm);
        }
        
        if (residualNorm < tolerance_) {
            return true;
        }
        
        // Compute Jacobian
        SparseMatrix J(totalDOF, totalDOF);
        computeJacobian(fields, residualFunc, J);
        
        // Solve linear system: J * dx = -F
        VectorX dx(totalDOF);
        bool converged = linearSolver_->solve(J, -F, dx);
        
        if (!converged) {
            logger_->warn("Linear solver failed in nonlinear iteration {}", iter);
            return false;
        }
        
        // Line search
        Real alpha = lineSearch(fields, dx, residualFunc, residualNorm);
        
        // Update solution
        x += alpha * dx;
        unpackFields(x, fields);
        
        // Update boundary conditions
        for (auto& field : fields) {
            field.updateBoundaryConditions();
        }
    }
    
    logger_->warn("Nonlinear solver did not converge in {} iterations", 
                 maxIterations_);
    return false;
}

void NonlinearSolver::computeJacobian(const std::vector<ScalarField>& fields,
                                     ResidualFunction residualFunc,
                                     SparseMatrix& J) {
    const Index numFields = fields.size();
    const Index numCells = mesh_->numCells();
    const Real epsilon = std::sqrt(EPSILON);
    
    std::vector<Triplet> triplets;
    triplets.reserve(numFields * numCells * 7); // Estimate
    
    // Finite difference Jacobian
    // (Full implementation would use automatic differentiation)
    
    std::vector<ScalarField> fields_plus = fields;
    std::vector<ScalarField> fields_minus = fields;
    
    for (Index fieldIdx = 0; fieldIdx < numFields; ++fieldIdx) {
        for (Index cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            Index colIdx = fieldIdx * numCells + cellIdx;
            
            // Perturb field
            Real orig = fields[fieldIdx][cellIdx];
            Real h = epsilon * (1.0 + std::abs(orig));
            
            fields_plus[fieldIdx][cellIdx] = orig + h;
            fields_minus[fieldIdx][cellIdx] = orig - h;
            
            // Compute perturbed residuals
            auto res_plus = residualFunc(fields_plus);
            auto res_minus = residualFunc(fields_minus);
            
            // Compute derivatives
            for (Index resFieldIdx = 0; resFieldIdx < numFields; ++resFieldIdx) {
                for (Index resCellIdx = 0; resCellIdx < numCells; ++resCellIdx) {
                    Index rowIdx = resFieldIdx * numCells + resCellIdx;
                    
                    Real deriv = (res_plus[resFieldIdx][resCellIdx] - 
                                 res_minus[resFieldIdx][resCellIdx]) / (2 * h);
                    
                    if (std::abs(deriv) > SMALL) {
                        triplets.emplace_back(rowIdx, colIdx, deriv);
                    }
                }
            }
            
            // Restore original value
            fields_plus[fieldIdx][cellIdx] = orig;
            fields_minus[fieldIdx][cellIdx] = orig;
        }
    }
    
    J.setFromTriplets(triplets.begin(), triplets.end());
}

Real NonlinearSolver::lineSearch(const std::vector<ScalarField>& fields,
                                const VectorX& dx,
                                ResidualFunction residualFunc,
                                Real f0) {
    Real alpha = 1.0;
    const Real c = 1e-4; // Armijo constant
    const Real rho = 0.5; // Backtracking factor
    const int maxLineSearch = 20;
    
    std::vector<ScalarField> fields_new = fields;
    VectorX x0(dx.size());
    packFields(fields, x0);
    
    for (int i = 0; i < maxLineSearch; ++i) {
        // Try new point
        VectorX x_new = x0 + alpha * dx;
        unpackFields(x_new, fields_new);
        
        // Evaluate residual
        auto residual = residualFunc(fields_new);
        VectorX F(dx.size());
        packFields(residual, F);
        Real f_new = 0.5 * F.squaredNorm();
        
        // Check Armijo condition
        if (f_new <= f0 + c * alpha * (-f0)) { // Approximate gradient
            return alpha;
        }
        
        alpha *= rho;
    }
    
    return alpha;
}

void NonlinearSolver::packFields(const std::vector<ScalarField>& fields,
                                VectorX& x) {
    Index idx = 0;
    for (const auto& field : fields) {
        for (Index i = 0; i < field.size(); ++i) {
            x[idx++] = field[i];
        }
    }
}

void NonlinearSolver::unpackFields(const VectorX& x,
                                  std::vector<ScalarField>& fields) {
    Index idx = 0;
    for (auto& field : fields) {
        for (Index i = 0; i < field.size(); ++i) {
            field[i] = x[idx++];
        }
    }
}
