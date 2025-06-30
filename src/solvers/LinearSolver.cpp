// ===== SOLVER IMPLEMENTATIONS =====

// === src/solvers/LinearSolver.cpp ===
#include "cfd/solvers/LinearSolver.hpp"
#include "cfd/parallel/Communication.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace cfd::solvers {

SharedPtr<LinearSolver> LinearSolver::create(LinearSolverType type,
                                           const SolverSettings& settings) {
    switch (type) {
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

// Base solver implementation
LinearSolver::LinearSolver(const SolverSettings& settings)
    : settings_(settings), iterations_(0), finalResidual_(0.0) {
}

void LinearSolver::setPreconditioner(SharedPtr<Preconditioner> precond) {
    preconditioner_ = precond;
}

bool LinearSolver::checkConvergence(Real residual, Real residual0) const {
    Real relResidual = residual / (residual0 + SMALL);
    
    if (settings_.verbose && iterations_ % settings_.printFrequency == 0) {
        logger_->info("  Iteration {}: residual = {:.3e}, relative = {:.3e}",
                     iterations_, residual, relResidual);
    }
    
    return (residual < settings_.tolerance) || 
           (relResidual < settings_.relativeTolerance);
}

// GMRES solver
GMRESSolver::GMRESSolver(const SolverSettings& settings)
    : LinearSolver(settings) {
    restart_ = settings.restartInterval > 0 ? settings.restartInterval : 30;
}

bool GMRESSolver::solve(const SparseMatrix& A, const VectorX& b, VectorX& x) {
    const Index n = b.size();
    const Index m = std::min(restart_, n);
    
    iterations_ = 0;
    
    // Workspace
    MatrixX V(n, m + 1);           // Arnoldi vectors
    MatrixX H(m + 1, m);           // Hessenberg matrix
    VectorX g(m + 1);              // Residual vector in Krylov space
    VectorX c(m + 1), s(m + 1);    // Givens rotation coefficients
    VectorX y(m);                  // Solution in Krylov space
    
    // Initial residual
    VectorX r = b - A * x;
    
    if (preconditioner_) {
        r = preconditioner_->apply(r);
    }
    
    Real beta = r.norm();
    Real residual0 = beta;
    
    if (checkConvergence(beta, residual0)) {
        finalResidual_ = beta;
        return true;
    }
    
    // GMRES with restarts
    while (iterations_ < settings_.maxIterations) {
        V.col(0) = r / beta;
        g.setZero();
        g(0) = beta;
        
        // Arnoldi process
        Index j;
        for (j = 0; j < m && iterations_ < settings_.maxIterations; ++j) {
            iterations_++;
            
            // Matrix-vector product
            VectorX w = A * V.col(j);
            
            if (preconditioner_) {
                w = preconditioner_->apply(w);
            }
            
            // Modified Gram-Schmidt
            for (Index i = 0; i <= j; ++i) {
                H(i, j) = w.dot(V.col(i));
                w -= H(i, j) * V.col(i);
            }
            
            H(j + 1, j) = w.norm();
            
            // Check for breakdown
            if (H(j + 1, j) < SMALL) {
                break;
            }
            
            V.col(j + 1) = w / H(j + 1, j);
            
            // Apply previous Givens rotations
            for (Index i = 0; i < j; ++i) {
                Real temp = c(i) * H(i, j) + s(i) * H(i + 1, j);
                H(i + 1, j) = -s(i) * H(i, j) + c(i) * H(i + 1, j);
                H(i, j) = temp;
            }
            
            // Compute new Givens rotation
            Real h_jj = H(j, j);
            Real h_j1j = H(j + 1, j);
            Real r_j = std::sqrt(h_jj * h_jj + h_j1j * h_j1j);
            
            c(j) = h_jj / r_j;
            s(j) = h_j1j / r_j;
            
            H(j, j) = r_j;
            H(j + 1, j) = 0.0;
            
            // Update residual
            g(j + 1) = -s(j) * g(j);
            g(j) = c(j) * g(j);
            
            beta = std::abs(g(j + 1));
            
            if (checkConvergence(beta, residual0)) {
                j++;
                break;
            }
        }
        
        // Solve upper triangular system
        for (Index i = j - 1; i >= 0; --i) {
            y(i) = g(i);
            for (Index k = i + 1; k < j; ++k) {
                y(i) -= H(i, k) * y(k);
            }
            y(i) /= H(i, i);
        }
        
        // Update solution
        for (Index i = 0; i < j; ++i) {
            x += y(i) * V.col(i);
        }
        
        // Check convergence
        if (beta < settings_.tolerance || 
            beta / residual0 < settings_.relativeTolerance) {
            finalResidual_ = beta;
            return true;
        }
        
        // Compute new residual for restart
        r = b - A * x;
        if (preconditioner_) {
            r = preconditioner_->apply(r);
        }
        beta = r.norm();
    }
    
    finalResidual_ = beta;
    return false;
}

// BiCGSTAB solver
bool BiCGSTABSolver::solve(const SparseMatrix& A, const VectorX& b, VectorX& x) {
    const Index n = b.size();
    iterations_ = 0;
    
    // Initial residual
    VectorX r = b - A * x;
    VectorX r0 = r;  // Shadow residual
    
    Real residual0 = r.norm();
    if (checkConvergence(residual0, residual0)) {
        finalResidual_ = residual0;
        return true;
    }
    
    // Initialize vectors
    VectorX p = r;
    VectorX v = VectorX::Zero(n);
    VectorX s = VectorX::Zero(n);
    VectorX t = VectorX::Zero(n);
    
    Real rho = 1.0, alpha = 1.0, omega = 1.0;
    Real rho_old = 1.0;
    
    // BiCGSTAB iteration
    while (iterations_ < settings_.maxIterations) {
        iterations_++;
        
        // rho = (r0, r)
        rho = r0.dot(r);
        
        // Check for breakdown
        if (std::abs(rho) < SMALL) {
            logger_->warn("BiCGSTAB breakdown: rho = {}", rho);
            break;
        }
        
        // beta = (rho / rho_old) * (alpha / omega)
        Real beta = (rho / rho_old) * (alpha / omega);
        
        // p = r + beta * (p - omega * v)
        p = r + beta * (p - omega * v);
        
        // Apply preconditioner
        VectorX p_hat = p;
        if (preconditioner_) {
            p_hat = preconditioner_->apply(p);
        }
        
        // v = A * p_hat
        v = A * p_hat;
        
        // alpha = rho / (r0, v)
        alpha = rho / r0.dot(v);
        
        // s = r - alpha * v
        s = r - alpha * v;
        
        // Check convergence
        Real s_norm = s.norm();
        if (checkConvergence(s_norm, residual0)) {
            x += alpha * p_hat;
            finalResidual_ = s_norm;
            return true;
        }
        
        // Apply preconditioner
        VectorX s_hat = s;
        if (preconditioner_) {
            s_hat = preconditioner_->apply(s);
        }
        
        // t = A * s_hat
        t = A * s_hat;
        
        // omega = (t, s) / (t, t)
        omega = t.dot(s) / t.dot(t);
        
        // Update solution
        x += alpha * p_hat + omega * s_hat;
        
        // Update residual
        r = s - omega * t;
        
        // Check convergence
        Real residual = r.norm();
        if (checkConvergence(residual, residual0)) {
            finalResidual_ = residual;
            return true;
        }
        
        // Check for breakdown
        if (std::abs(omega) < SMALL) {
            logger_->warn("BiCGSTAB breakdown: omega = {}", omega);
            break;
        }
        
        rho_old = rho;
    }
    
    finalResidual_ = r.norm();
    return false;
}

// Conjugate Gradient solver
bool CGSolver::solve(const SparseMatrix& A, const VectorX& b, VectorX& x) {
    iterations_ = 0;
    
    // Initial residual
    VectorX r = b - A * x;
    VectorX z = r;
    
    if (preconditioner_) {
        z = preconditioner_->apply(r);
    }
    
    VectorX p = z;
    
    Real rz_old = r.dot(z);
    Real residual0 = r.norm();
    
    if (checkConvergence(residual0, residual0)) {
        finalResidual_ = residual0;
        return true;
    }
    
    // CG iteration
    while (iterations_ < settings_.maxIterations) {
        iterations_++;
        
        // q = A * p
        VectorX q = A * p;
        
        // alpha = (r, z) / (p, q)
        Real alpha = rz_old / p.dot(q);
        
        // x = x + alpha * p
        x += alpha * p;
        
        // r = r - alpha * q
        r -= alpha * q;
        
        // Check convergence
        Real residual = r.norm();
        if (checkConvergence(residual, residual0)) {
            finalResidual_ = residual;
            return true;
        }
        
        // z = M^{-1} * r
        z = r;
        if (preconditioner_) {
            z = preconditioner_->apply(r);
        }
        
        Real rz_new = r.dot(z);
        
        // beta = (r_new, z_new) / (r_old, z_old)
        Real beta = rz_new / rz_old;
        
        // p = z + beta * p
        p = z + beta * p;
        
        rz_old = rz_new;
    }
    
    finalResidual_ = r.norm();
    return false;
}

// Direct solver (using Eigen's SparseLU)
bool DirectSolver::solve(const SparseMatrix& A, const VectorX& b, VectorX& x) {
    Eigen::SparseLU<SparseMatrix> solver;
    
    solver.analyzePattern(A);
    solver.factorize(A);
    
    if (solver.info() != Eigen::Success) {
        logger_->error("Direct solver factorization failed");
        return false;
    }
    
    x = solver.solve(b);
    
    if (solver.info() != Eigen::Success) {
        logger_->error("Direct solver solve failed");
        return false;
    }
    
    // Compute residual
    VectorX r = b - A * x;
    finalResidual_ = r.norm();
    iterations_ = 1;
    
    return finalResidual_ < settings_.tolerance;
}

// Preconditioner implementations
SharedPtr<Preconditioner> Preconditioner::create(PreconditionerType type) {
    switch (type) {
        case PreconditionerType::NONE:
            return std::make_shared<NoPreconditioner>();
        case PreconditionerType::JACOBI:
            return std::make_shared<JacobiPreconditioner>();
        case PreconditionerType::ILU0:
            return std::make_shared<ILU0Preconditioner>();
        case PreconditionerType::GAUSS_SEIDEL:
            return std::make_shared<GaussSeidelPreconditioner>();
        default:
            throw std::runtime_error("Unknown preconditioner type");
    }
}

// Jacobi preconditioner
void JacobiPreconditioner::setup(const SparseMatrix& A) {
    diagonal_ = A.diagonal();
    
    // Invert diagonal
    for (Index i = 0; i < diagonal_.size(); ++i) {
        if (std::abs(diagonal_[i]) > SMALL) {
            diagonal_[i] = 1.0 / diagonal_[i];
        } else {
            diagonal_[i] = 1.0;
        }
    }
}

VectorX JacobiPreconditioner::apply(const VectorX& r) const {
    return diagonal_.cwiseProduct(r);
}

// ILU(0) preconditioner
void ILU0Preconditioner::setup(const SparseMatrix& A) {
    L_ = A.triangularView<Eigen::Lower>();
    U_ = A.triangularView<Eigen::Upper>();
    
    const Index n = A.rows();
    
    // ILU(0) factorization
    for (Index k = 1; k < n; ++k) {
        for (SparseMatrix::InnerIterator it(L_, k); it && it.row() < k; ++it) {
            Index i = it.row();
            Real L_ik = it.value();
            
            // Update L
            L_ik /= U_.coeff(i, i);
            L_.coeffRef(k, i) = L_ik;
            
            // Update U
            for (SparseMatrix::InnerIterator jt(U_, i); jt && jt.col() > i; ++jt) {
                Index j = jt.col();
                if (A.coeff(k, j) != 0) {
                    U_.coeffRef(k, j) -= L_ik * jt.value();
                }
            }
        }
    }
}

VectorX ILU0Preconditioner::apply(const VectorX& r) const {
    // Forward substitution: L * y = r
    VectorX y = L_.triangularView<Eigen::Lower>().solve(r);
    
    // Backward substitution: U * x = y
    return U_.triangularView<Eigen::Upper>().solve(y);
}
