// ===== PHYSICS MODELS IMPLEMENTATIONS =====

// === src/physics/NavierStokes.cpp ===
#include "cfd/physics/NavierStokes.hpp"
#include "cfd/numerics/GradientScheme.hpp"
#include <cmath>

namespace cfd::physics {

NavierStokes::NavierStokes(const PhysicsParameters& params)
    : params_(params) {
    
    // Validate parameters
    if (params_.rho0 <= 0) {
        throw std::runtime_error("Invalid reference density");
    }
    if (params_.mu0 <= 0) {
        throw std::runtime_error("Invalid reference viscosity");
    }
}

// Convection term: div(rho*U*U)
VectorField NavierStokes::convection(const VectorField& U,
                                     const ScalarField& rho) const {
    auto mesh = U.mesh();
    VectorField conv(mesh, "convection");
    conv = Vector3::Zero();
    
    // Compute flux through faces
    for (Index faceId = 0; faceId < mesh->numFaces(); ++faceId) {
        const Face& face = mesh->face(faceId);
        
        // Interpolate velocity to face
        Vector3 Uf;
        Real rhof;
        
        if (face.isBoundary()) {
            Uf = U[face.owner()];
            rhof = rho[face.owner()];
        } else {
            Real w = face.interpolationWeight();
            Uf = w * U[face.owner()] + (1 - w) * U[face.neighbor()];
            rhof = w * rho[face.owner()] + (1 - w) * rho[face.neighbor()];
        }
        
        // Mass flux
        Real phi = rhof * Uf.dot(face.normal()) * face.area();
        
        // Add to owner cell
        conv[face.owner()] += phi * Uf / mesh->cell(face.owner()).volume();
        
        // Subtract from neighbor cell
        if (!face.isBoundary()) {
            conv[face.neighbor()] -= phi * Uf / mesh->cell(face.neighbor()).volume();
        }
    }
    
    return conv;
}

// Diffusion term: div(mu*grad(U) + mu*(grad(U))^T)
VectorField NavierStokes::diffusion(const VectorField& U,
                                    const ScalarField& mu,
                                    const ScalarField* nut) const {
    auto mesh = U.mesh();
    VectorField diff(mesh, "diffusion");
    diff = Vector3::Zero();
    
    // Compute velocity gradient
    numerics::GreenGaussGradient gradScheme(mesh);
    TensorField gradU = gradScheme.gradient(U);
    
    // Effective viscosity
    ScalarField muEff = mu;
    if (nut) {
        for (Index i = 0; i < mesh->numCells(); ++i) {
            muEff[i] = mu[i] + (*nut)[i];
        }
    }
    
    // Compute diffusive flux through faces
    for (Index faceId = 0; faceId < mesh->numFaces(); ++faceId) {
        const Face& face = mesh->face(faceId);
        Index owner = face.owner();
        
        // Interpolate gradient to face
        Matrix3 gradUf;
        Real muEfff;
        
        if (face.isBoundary()) {
            gradUf = gradU[owner];
            muEfff = muEff[owner];
        } else {
            Index neighbor = face.neighbor();
            Real w = face.interpolationWeight();
            gradUf = w * gradU[owner] + (1 - w) * gradU[neighbor];
            muEfff = w * muEff[owner] + (1 - w) * muEff[neighbor];
        }
        
        // Stress tensor: tau = mu*(grad(U) + grad(U)^T) - (2/3)*mu*div(U)*I
        Matrix3 tau = muEfff * (gradUf + gradUf.transpose());
        
        // Add bulk viscosity term if compressible
        if (params_.compressible) {
            Real divU = gradUf.trace();
            tau -= (2.0/3.0) * muEfff * divU * Matrix3::Identity();
        }
        
        // Viscous flux
        Vector3 flux = tau * face.normal() * face.area();
        
        // Add to owner cell
        diff[owner] += flux / mesh->cell(owner).volume();
        
        // Subtract from neighbor cell
        if (!face.isBoundary()) {
            Index neighbor = face.neighbor();
            diff[neighbor] -= flux / mesh->cell(neighbor).volume();
        }
    }
    
    return diff;
}

// Pressure gradient term
VectorField NavierStokes::pressureGradient(const ScalarField& p,
                                          const ScalarField& rho) const {
    auto mesh = p.mesh();
    VectorField gradP(mesh, "gradP");
    
    // Use Green-Gauss gradient
    numerics::GreenGaussGradient gradScheme(mesh);
    gradP = gradScheme.gradient(p);
    
    // Divide by density for momentum equation
    for (Index i = 0; i < mesh->numCells(); ++i) {
        gradP[i] /= rho[i];
    }
    
    return gradP;
}

// Viscosity models
Real NavierStokes::viscosity(Real T) const {
    if (params_.viscosityModel == ViscosityModel::CONSTANT) {
        return params_.mu0;
    } else if (params_.viscosityModel == ViscosityModel::SUTHERLAND) {
        // Sutherland's law
        Real T0 = params_.T0;
        Real S = 110.4; // Sutherland's constant for air
        return params_.mu0 * std::pow(T/T0, 1.5) * (T0 + S) / (T + S);
    } else if (params_.viscosityModel == ViscosityModel::POWER_LAW) {
        // Power law: mu = mu0 * (T/T0)^n
        return params_.mu0 * std::pow(T/params_.T0, params_.viscosityExponent);
    }
    
    return params_.mu0;
}

// Source terms
VectorField NavierStokes::sourceTerms(const VectorField& U,
                                     const ScalarField& rho) const {
    auto mesh = U.mesh();
    VectorField source(mesh, "source");
    
    // Gravity/buoyancy
    if (params_.gravity.norm() > 0) {
        for (Index i = 0; i < mesh->numCells(); ++i) {
            source[i] = rho[i] * params_.gravity;
        }
    } else {
        source = Vector3::Zero();
    }
    
    // Add other source terms (MRF, porous media, etc.)
    
    return source;
}

// Energy equation terms
ScalarField NavierStokes::energyConvection(const VectorField& U,
                                          const ScalarField& T,
                                          const ScalarField& rho,
                                          const ScalarField& Cp) const {
    auto mesh = T.mesh();
    ScalarField conv(mesh, "energyConvection");
    conv = 0.0;
    
    // Compute enthalpy flux through faces
    for (Index faceId = 0; faceId < mesh->numFaces(); ++faceId) {
        const Face& face = mesh->face(faceId);
        
        // Interpolate values to face
        Vector3 Uf;
        Real Tf, rhof, Cpf;
        
        if (face.isBoundary()) {
            Uf = U[face.owner()];
            Tf = T[face.owner()];
            rhof = rho[face.owner()];
            Cpf = Cp[face.owner()];
        } else {
            Real w = face.interpolationWeight();
            Uf = w * U[face.owner()] + (1 - w) * U[face.neighbor()];
            Tf = w * T[face.owner()] + (1 - w) * T[face.neighbor()];
            rhof = w * rho[face.owner()] + (1 - w) * rho[face.neighbor()];
            Cpf = w * Cp[face.owner()] + (1 - w) * Cp[face.neighbor()];
        }
        
        // Enthalpy flux
        Real phi = rhof * Cpf * Tf * Uf.dot(face.normal()) * face.area();
        
        // Add to cells
        conv[face.owner()] += phi / mesh->cell(face.owner()).volume();
        if (!face.isBoundary()) {
            conv[face.neighbor()] -= phi / mesh->cell(face.neighbor()).volume();
        }
    }
    
    return conv;
}

ScalarField NavierStokes::energyDiffusion(const VectorField& U,
                                         const ScalarField& T,
                                         const ScalarField& k,
                                         const ScalarField* alphat) const {
    auto mesh = T.mesh();
    ScalarField diff(mesh, "energyDiffusion");
    diff = 0.0;
    
    // Effective thermal diffusivity
    ScalarField alphaEff = k;
    if (alphat) {
        for (Index i = 0; i < mesh->numCells(); ++i) {
            alphaEff[i] = k[i] + (*alphat)[i];
        }
    }
    
    // Temperature gradient
    numerics::GreenGaussGradient gradScheme(mesh);
    VectorField gradT = gradScheme.gradient(T);
    
    // Compute diffusive flux
    for (Index faceId = 0; faceId < mesh->numFaces(); ++faceId) {
        const Face& face = mesh->face(faceId);
        
        // Interpolate to face
        Vector3 gradTf;
        Real alphaEfff;
        
        if (face.isBoundary()) {
            gradTf = gradT[face.owner()];
            alphaEfff = alphaEff[face.owner()];
        } else {
            Real w = face.interpolationWeight();
            gradTf = w * gradT[face.owner()] + (1 - w) * gradT[face.neighbor()];
            alphaEfff = w * alphaEff[face.owner()] + (1 - w) * alphaEff[face.neighbor()];
        }
        
        // Heat flux
        Real flux = alphaEfff * gradTf.dot(face.normal()) * face.area();
        
        // Add to cells
        diff[face.owner()] += flux / mesh->cell(face.owner()).volume();
        if (!face.isBoundary()) {
            diff[face.neighbor()] -= flux / mesh->cell(face.neighbor()).volume();
        }
    }
    
    return diff;
}

ScalarField NavierStokes::viscousDissipation(const VectorField& U,
                                            const ScalarField& mu) const {
    auto mesh = U.mesh();
    ScalarField dissipation(mesh, "viscousDissipation");
    
    // Compute velocity gradient
    numerics::GreenGaussGradient gradScheme(mesh);
    TensorField gradU = gradScheme.gradient(U);
    
    for (Index i = 0; i < mesh->numCells(); ++i) {
        const Matrix3& grad = gradU[i];
        
        // Strain rate tensor: S = 0.5*(grad(U) + grad(U)^T)
        Matrix3 S = 0.5 * (grad + grad.transpose());
        
        // Dissipation: Phi = 2*mu*S:S
        dissipation[i] = 2 * mu[i] * S.squaredNorm();
        
        // Add bulk viscosity contribution if compressible
        if (params_.compressible) {
            Real divU = grad.trace();
            dissipation[i] -= (2.0/3.0) * mu[i] * divU * divU;
        }
    }
    
    return dissipation;
}

// Compressible flow specific
Real NavierStokes::pressure(Real rho, Real e) const {
    // Ideal gas: p = (gamma - 1) * rho * e
    return (params_.gamma - 1) * rho * e;
}

Real NavierStokes::soundSpeed(Real p, Real rho) const {
    return std::sqrt(params_.gamma * p / rho);
}

Real NavierStokes::internalEnergy(Real p, Real rho) const {
    return p / ((params_.gamma - 1) * rho);
}

// Compressible state methods
Real CompressibleState::density() const {
    return conserved[0];
}

Vector3 CompressibleState::velocity() const {
    Real rho = conserved[0];
    return Vector3(conserved[1]/rho, conserved[2]/rho, conserved[3]/rho);
}

Real CompressibleState::pressure() const {
    Real rho = conserved[0];
    Vector3 U = velocity();
    Real E = conserved[4];
    Real gamma = 1.4; // Should be passed from physics
    
    return (gamma - 1) * (E - 0.5 * rho * U.squaredNorm());
}

Real CompressibleState::totalEnthalpy() const {
    return (conserved[4] + pressure()) / conserved[0];
}

Real CompressibleState::soundSpeed() const {
    Real gamma = 1.4;
    return std::sqrt(gamma * pressure() / density());
}

VectorX CompressibleState::flux(const Vector3& normal) const {
    VectorX F(5);
    
    Real rho = density();
    Vector3 U = velocity();
    Real p = pressure();
    Real E = conserved[4];
    Real un = U.dot(normal);
    
    F[0] = rho * un;
    F[1] = rho * un * U.x() + p * normal.x();
    F[2] = rho * un * U.y() + p * normal.y();
    F[3] = rho * un * U.z() + p * normal.z();
    F[4] = (E + p) * un;
    
    return F;
}
