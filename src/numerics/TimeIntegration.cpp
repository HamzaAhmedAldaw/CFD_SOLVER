
// === src/numerics/TimeIntegration.cpp ===
#include "cfd/numerics/TimeIntegration.hpp"
#include "cfd/solvers/LinearSolver.hpp"

namespace cfd::numerics {

SharedPtr<TimeIntegration> TimeIntegration::create(TimeScheme scheme,
                                                  SharedPtr<Mesh> mesh) {
    switch (scheme) {
        case TimeScheme::EULER_EXPLICIT:
            return std::make_shared<ExplicitEuler>(mesh);
        case TimeScheme::EULER_IMPLICIT:
            return std::make_shared<ImplicitEuler>(mesh);
        case TimeScheme::BDF2:
            return std::make_shared<BDF2>(mesh);
        case TimeScheme::BDF3:
            return std::make_shared<BDF3>(mesh);
        case TimeScheme::RK4:
            return std::make_shared<RungeKutta4>(mesh);
        default:
            throw std::runtime_error("Unknown time integration scheme");
    }
}

// Explicit Euler
void ExplicitEuler::advance(ScalarField& field,
                           const std::function<ScalarField(const ScalarField&)>& residual,
                           Real dt) {
    ScalarField R = residual(field);
    
    for (Index i = 0; i < field.size(); ++i) {
        field[i] += dt * R[i];
    }
    
    field.updateBoundaryConditions();
}

void ExplicitEuler::advance(VectorField& field,
                           const std::function<VectorField(const VectorField&)>& residual,
                           Real dt) {
    VectorField R = residual(field);
    
    for (Index i = 0; i < field.size(); ++i) {
        field[i] += dt * R[i];
    }
    
    field.updateBoundaryConditions();
}

// Implicit Euler
void ImplicitEuler::advance(ScalarField& field,
                           const std::function<ScalarField(const ScalarField&)>& residual,
                           Real dt) {
    // Solve: (I - dt*J) * delta = dt * R(phi)
    // where J is Jacobian of residual
    
    const Index n = field.size();
    SparseMatrix A(n, n);
    VectorX b(n);
    VectorX x(n);
    
    // Build system using finite differences for Jacobian
    // (Full implementation would use automatic differentiation)
    
    // Simplified: use explicit update with under-relaxation
    ScalarField R = residual(field);
    Real alpha = 0.5; // Under-relaxation
    
    for (Index i = 0; i < n; ++i) {
        field[i] += alpha * dt * R[i];
    }
    
    field.updateBoundaryConditions();
}

// BDF2 (Backward Differentiation Formula, 2nd order)
void BDF2::advance(ScalarField& field,
                  const std::function<ScalarField(const ScalarField&)>& residual,
                  Real dt) {
    if (!initialized_) {
        // First time step: use implicit Euler
        ImplicitEuler euler(mesh_);
        euler.advance(field, residual, dt);
        initialized_ = true;
        return;
    }
    
    // BDF2: (3*phi^{n+1} - 4*phi^n + phi^{n-1})/(2*dt) = R(phi^{n+1})
    
    const ScalarField& phiOld = field.oldTime();
    const ScalarField& phiOldOld = field.oldOldTime();
    
    // Newton iteration (simplified)
    ScalarField phiNew = field;
    
    for (int iter = 0; iter < maxIterations_; ++iter) {
        ScalarField R = residual(phiNew);
        ScalarField lhs = (3*phiNew - 4*phiOld + phiOldOld) / (2*dt);
        ScalarField residualNewton = lhs - R;
        
        // Check convergence
        Real resNorm = 0.0;
        for (Index i = 0; i < field.size(); ++i) {
            resNorm += residualNewton[i] * residualNewton[i];
        }
        resNorm = std::sqrt(resNorm / field.size());
        
        if (resNorm < tolerance_) break;
        
        // Update (simplified - should solve linear system)
        for (Index i = 0; i < field.size(); ++i) {
            phiNew[i] -= 0.5 * residualNewton[i]; // Under-relaxation
        }
    }
    
    field = phiNew;
    field.updateBoundaryConditions();
}

// Runge-Kutta 4th order
void RungeKutta4::advance(ScalarField& field,
                         const std::function<ScalarField(const ScalarField&)>& residual,
                         Real dt) {
    ScalarField phi0 = field;
    
    // Stage 1
    ScalarField k1 = residual(phi0);
    ScalarField phi1 = phi0 + (dt/2) * k1;
    phi1.updateBoundaryConditions();
    
    // Stage 2
    ScalarField k2 = residual(phi1);
    ScalarField phi2 = phi0 + (dt/2) * k2;
    phi2.updateBoundaryConditions();
    
    // Stage 3
    ScalarField k3 = residual(phi2);
    ScalarField phi3 = phi0 + dt * k3;
    phi3.updateBoundaryConditions();
    
    // Stage 4
    ScalarField k4 = residual(phi3);
    
    // Final update
    for (Index i = 0; i < field.size(); ++i) {
        field[i] = phi0[i] + (dt/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }
    
    field.updateBoundaryConditions();
}

void RungeKutta4::advance(VectorField& field,
                         const std::function<VectorField(const VectorField&)>& residual,
                         Real dt) {
    VectorField phi0 = field;
    
    // Stage 1
    VectorField k1 = residual(phi0);
    VectorField phi1 = phi0 + (dt/2) * k1;
    phi1.updateBoundaryConditions();
    
    // Stage 2
    VectorField k2 = residual(phi1);
    VectorField phi2 = phi0 + (dt/2) * k2;
    phi2.updateBoundaryConditions();
    
    // Stage 3
    VectorField k3 = residual(phi2);
    VectorField phi3 = phi0 + dt * k3;
    phi3.updateBoundaryConditions();
    
    // Stage 4
    VectorField k4 = residual(phi3);
    
    // Final update
    for (Index i = 0; i < field.size(); ++i) {
        field[i] = phi0[i] + (dt/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }
    
    field.updateBoundaryConditions();
}

// Crank-Nicolson (for diffusion-dominated problems)
void CrankNicolson::advance(ScalarField& field,
                           const std::function<ScalarField(const ScalarField&)>& residual,
                           Real dt) {
    // Crank-Nicolson: (phi^{n+1} - phi^n)/dt = 0.5*(R(phi^{n+1}) + R(phi^n))
    
    ScalarField phiOld = field;
    ScalarField Rold = residual(phiOld);
    
    // Newton iteration
    for (int iter = 0; iter < maxIterations_; ++iter) {
        ScalarField Rnew = residual(field);
        ScalarField lhs = (field - phiOld) / dt;
        ScalarField rhs = 0.5 * (Rnew + Rold);
        ScalarField residualNewton = lhs - rhs;
        
        // Check convergence
        Real resNorm = 0.0;
        for (Index i = 0; i < field.size(); ++i) {
            resNorm += residualNewton[i] * residualNewton[i];
        }
        resNorm = std::sqrt(resNorm / field.size());
        
        if (resNorm < tolerance_) break;
        
        // Update (simplified)
        for (Index i = 0; i < field.size(); ++i) {
            field[i] -= 0.5 * residualNewton[i];
        }
    }
    
    field.updateBoundaryConditions();
}

} // namespace cfd::numerics
