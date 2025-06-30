// === src/physics/TurbulenceModel.cpp ===
#include "cfd/physics/TurbulenceModel.hpp"
#include "cfd/numerics/GradientScheme.hpp"
#include <algorithm>
#include <cmath>

namespace cfd::physics {

SharedPtr<TurbulenceModel> TurbulenceModel::create(
    TurbulenceModelType type,
    SharedPtr<Mesh> mesh,
    const PhysicsParameters& params) {
    
    switch (type) {
        case TurbulenceModelType::SPALART_ALLMARAS:
            return std::make_shared<SpalartAllmaras>(mesh, params);
        case TurbulenceModelType::K_EPSILON:
            return std::make_shared<KEpsilon>(mesh, params);
        case TurbulenceModelType::K_OMEGA:
            return std::make_shared<KOmega>(mesh, params);
        case TurbulenceModelType::K_OMEGA_SST:
            return std::make_shared<KOmegaSST>(mesh, params);
        default:
            throw std::runtime_error("Unknown turbulence model type");
    }
}

// Spalart-Allmaras implementation
SpalartAllmaras::SpalartAllmaras(SharedPtr<Mesh> mesh,
                               const PhysicsParameters& params)
    : TurbulenceModel(mesh, params) {
    
    // Model constants
    sigma_ = 2.0/3.0;
    Cb1_ = 0.1355;
    Cb2_ = 0.622;
    Cv1_ = 7.1;
    Cw1_ = Cb1_ / (kappa_ * kappa_) + (1 + Cb2_) / sigma_;
    Cw2_ = 0.3;
    Cw3_ = 2.0;
    
    // Create fields
    nuTilde_ = std::make_shared<ScalarField>(mesh, "nuTilde");
}

void SpalartAllmaras::solve(const VectorField& U,
                          ScalarField& nut,
                          Real dt) {
    // Compute velocity gradient magnitude
    numerics::GreenGaussGradient gradScheme(mesh_);
    TensorField gradU = gradScheme.gradient(U);
    
    ScalarField S(mesh_, "S"); // Strain rate magnitude
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        const Matrix3& grad = gradU[i];
        Matrix3 Sij = 0.5 * (grad + grad.transpose());
        S[i] = std::sqrt(2 * Sij.squaredNorm());
    }
    
    // Transport equation for nuTilde
    ScalarField residual(mesh_, "residual");
    
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        Real nu = params_.mu0 / params_.rho0;
        Real nuT = (*nuTilde_)[i];
        Real chi = nuT / nu;
        Real fv1 = chi*chi*chi / (chi*chi*chi + Cv1_*Cv1_*Cv1_);
        
        // Production term
        Real Stilde = S[i] + nuT / (kappa_ * kappa_ * dw_[i] * dw_[i]) * fv2(chi);
        Real production = Cb1_ * Stilde * nuT;
        
        // Destruction term
        Real r = std::min(nuT / (Stilde * kappa_ * kappa_ * dw_[i] * dw_[i]), 10.0);
        Real g = r + Cw2_ * (std::pow(r, 6) - r);
        Real fw = g * std::pow((1 + std::pow(Cw3_, 6)) / 
                              (std::pow(g, 6) + std::pow(Cw3_, 6)), 1.0/6.0);
        Real destruction = Cw1_ * fw * nuT * nuT / (dw_[i] * dw_[i]);
        
        residual[i] = production - destruction;
    }
    
    // Add convection and diffusion terms
    VectorField gradNuTilde = gradScheme.gradient(*nuTilde_);
    
    // Update nuTilde
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        (*nuTilde_)[i] += dt * residual[i];
        (*nuTilde_)[i] = std::max((*nuTilde_)[i], Real(0));
    }
    
    // Update turbulent viscosity
    updateNut(*nuTilde_, nullptr, nut);
}

void SpalartAllmaras::updateNut(const ScalarField& k,
                              const ScalarField* omega,
                              ScalarField& nut) {
    Real nu = params_.mu0 / params_.rho0;
    
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        Real chi = (*nuTilde_)[i] / nu;
        Real fv1 = chi*chi*chi / (chi*chi*chi + Cv1_*Cv1_*Cv1_);
        nut[i] = (*nuTilde_)[i] * fv1 * params_.rho0;
    }
}

Real SpalartAllmaras::fv2(Real chi) const {
    return 1.0 - chi / (1 + chi * fv1(chi));
}

Real SpalartAllmaras::fv1(Real chi) const {
    return chi*chi*chi / (chi*chi*chi + Cv1_*Cv1_*Cv1_);
}

// k-epsilon implementation
KEpsilon::KEpsilon(SharedPtr<Mesh> mesh, const PhysicsParameters& params)
    : TurbulenceModel(mesh, params) {
    
    // Model constants
    Cmu_ = 0.09;
    C1e_ = 1.44;
    C2e_ = 1.92;
    sigmaK_ = 1.0;
    sigmaE_ = 1.3;
    
    // Create fields
    k_ = std::make_shared<ScalarField>(mesh, "k");
    epsilon_ = std::make_shared<ScalarField>(mesh, "epsilon");
}

void KEpsilon::solve(const VectorField& U, ScalarField& nut, Real dt) {
    // Compute production term
    ScalarField Pk = computeProduction(U);
    
    // k equation
    solveKEquation(U, Pk, nut, dt);
    
    // epsilon equation
    solveEpsilonEquation(U, Pk, nut, dt);
    
    // Update turbulent viscosity
    updateNut(*k_, epsilon_.get(), nut);
}

void KEpsilon::solveKEquation(const VectorField& U,
                             const ScalarField& Pk,
                             const ScalarField& nut,
                             Real dt) {
    // Simplified implementation
    // Full implementation would include convection and diffusion
    
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        Real production = Pk[i];
        Real dissipation = (*epsilon_)[i];
        
        (*k_)[i] += dt * (production - dissipation);
        (*k_)[i] = std::max((*k_)[i], Real(1e-10));
    }
}

void KEpsilon::solveEpsilonEquation(const VectorField& U,
                                   const ScalarField& Pk,
                                   const ScalarField& nut,
                                   Real dt) {
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        Real k = (*k_)[i];
        Real epsilon = (*epsilon_)[i];
        
        Real production = C1e_ * epsilon / k * Pk[i];
        Real dissipation = C2e_ * epsilon * epsilon / k;
        
        (*epsilon_)[i] += dt * (production - dissipation);
        (*epsilon_)[i] = std::max((*epsilon_)[i], Real(1e-10));
    }
}

ScalarField KEpsilon::computeProduction(const VectorField& U) {
    ScalarField Pk(mesh_, "Pk");
    
    // Compute velocity gradient
    numerics::GreenGaussGradient gradScheme(mesh_);
    TensorField gradU = gradScheme.gradient(U);
    
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        const Matrix3& grad = gradU[i];
        Matrix3 S = 0.5 * (grad + grad.transpose());
        
        // Production: Pk = 2*nut*S:S
        // Get nut from previous iteration
        Real nutValue = 0.0; // Should access current nut
        Pk[i] = 2 * nutValue * S.squaredNorm();
    }
    
    return Pk;
}

void KEpsilon::updateNut(const ScalarField& k,
                        const ScalarField* epsilon,
                        ScalarField& nut) {
    if (!epsilon) {
        throw std::runtime_error("k-epsilon model requires epsilon field");
    }
    
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        Real kValue = k[i];
        Real epsilonValue = (*epsilon)[i];
        
        nut[i] = params_.rho0 * Cmu_ * kValue * kValue / (epsilonValue + SMALL);
    }
}

// k-omega SST implementation
KOmegaSST::KOmegaSST(SharedPtr<Mesh> mesh, const PhysicsParameters& params)
    : TurbulenceModel(mesh, params) {
    
    // Model constants - Set 1 (k-omega)
    sigmaK1_ = 0.85;
    sigmaW1_ = 0.5;
    beta1_ = 0.075;
    betaStar_ = 0.09;
    a1_ = 0.31;
    gamma1_ = beta1_ / betaStar_ - sigmaW1_ * kappa_ * kappa_ / std::sqrt(betaStar_);
    
    // Model constants - Set 2 (k-epsilon)
    sigmaK2_ = 1.0;
    sigmaW2_ = 0.856;
    beta2_ = 0.0828;
    gamma2_ = beta2_ / betaStar_ - sigmaW2_ * kappa_ * kappa_ / std::sqrt(betaStar_);
    
    // Create fields
    k_ = std::make_shared<ScalarField>(mesh, "k");
    omega_ = std::make_shared<ScalarField>(mesh, "omega");
    F1_ = std::make_shared<ScalarField>(mesh, "F1");
    F2_ = std::make_shared<ScalarField>(mesh, "F2");
}

void KOmegaSST::solve(const VectorField& U, ScalarField& nut, Real dt) {
    // Update blending functions
    updateBlendingFunctions(U);
    
    // Compute production
    ScalarField Pk = computeProduction(U, nut);
    
    // Solve k equation
    solveKEquation(U, Pk, nut, dt);
    
    // Solve omega equation
    solveOmegaEquation(U, Pk, nut, dt);
    
    // Update turbulent viscosity
    updateNut(*k_, omega_.get(), nut);
}

void KOmegaSST::updateBlendingFunctions(const VectorField& U) {
    numerics::GreenGaussGradient gradScheme(mesh_);
    VectorField gradK = gradScheme.gradient(*k_);
    VectorField gradOmega = gradScheme.gradient(*omega_);
    
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        Real k = (*k_)[i];
        Real omega = (*omega_)[i];
        Real nu = params_.mu0 / params_.rho0;
        Real y = dw_[i]; // Wall distance
        
        // Cross-diffusion term
        Real CDkw = std::max(2 * params_.rho0 * sigmaW2_ / omega * 
                            gradK[i].dot(gradOmega[i]), Real(1e-10));
        
        // Blending function F1
        Real arg1_1 = std::sqrt(k) / (betaStar_ * omega * y);
        Real arg1_2 = 500 * nu / (y * y * omega);
        Real arg1_3 = 4 * params_.rho0 * sigmaW2_ * k / (CDkw * y * y);
        Real arg1 = std::min(std::max(arg1_1, arg1_2), arg1_3);
        (*F1_)[i] = std::tanh(arg1 * arg1 * arg1 * arg1);
        
        // Blending function F2
        Real arg2_1 = 2 * std::sqrt(k) / (betaStar_ * omega * y);
        Real arg2_2 = 500 * nu / (y * y * omega);
        Real arg2 = std::max(arg2_1, arg2_2);
        (*F2_)[i] = std::tanh(arg2 * arg2);
    }
}

void KOmegaSST::updateNut(const ScalarField& k,
                         const ScalarField* omega,
                         ScalarField& nut) {
    if (!omega) {
        throw std::runtime_error("k-omega SST model requires omega field");
    }
    
    // Compute strain rate magnitude
    numerics::GreenGaussGradient gradScheme(mesh_);
    TensorField gradU = gradScheme.gradient(U_); // Need to store U reference
    
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        const Matrix3& grad = gradU[i];
        Matrix3 S = 0.5 * (grad + grad.transpose());
        Real Smag = std::sqrt(2 * S.squaredNorm());
        
        // SST modification
        Real arg = std::min(k[i] / (betaStar_ * (*omega)[i]), 
                           a1_ * k[i] / ((*F2_)[i] * a1_ * Smag));
        
        nut[i] = params_.rho0 * arg;
    }
}

Real KOmegaSST::blend(Real val1, Real val2, Real F1) const {
    return F1 * val1 + (1 - F1) * val2;
}

// Wall functions
void TurbulenceModel::computeWallDistance() {
    // Simple implementation - distance to nearest wall
    // Full implementation would use Poisson equation or other methods
    
    dw_.resize(mesh_->numCells());
    
    for (Index cellId = 0; cellId < mesh_->numCells(); ++cellId) {
        Real minDist = 1e10;
        const Vector3& cellCenter = mesh_->cell(cellId).center();
        
        // Find distance to wall boundaries
        for (const auto& patch : mesh_->boundaryPatches()) {
            if (patch.type() == BCType::WALL) {
                for (const Face* face : patch.faces()) {
                    Real dist = (face->center() - cellCenter).norm();
                    minDist = std::min(minDist, dist);
                }
            }
        }
        
        dw_[cellId] = minDist;
    }
}

ScalarField TurbulenceModel::yPlus(const VectorField& U,
                                  const ScalarField& nu) const {
    ScalarField yplus(mesh_, "yPlus");
    
    // Compute friction velocity at walls
    for (const auto& patch : mesh_->boundaryPatches()) {
        if (patch.type() == BCType::WALL) {
            for (const Face* face : patch.faces()) {
                Index cellId = face->owner();
                
                // Wall shear stress
                Real y = dw_[cellId];
                Real uParallel = U[cellId].norm(); // Simplified
                Real nuValue = nu[cellId] / params_.rho0;
                
                // Newton iteration for friction velocity
                Real utau = std::sqrt(nuValue * uParallel / y);
                
                for (int iter = 0; iter < 5; ++iter) {
                    Real yp = y * utau / nuValue;
                    Real up = uParallel / utau;
                    
                    // Van Driest profile
                    Real upNew = std::log(1 + kappa_ * yp) / kappa_ + 
                                7.8 * (1 - std::exp(-yp/11) - yp/11 * std::exp(-yp/3));
                    
                    utau *= up / upNew;
                }
                
                yplus[cellId] = y * utau / nuValue;
            }
        }
    }
    
    return yplus;
}

} // namespace cfd::physics
