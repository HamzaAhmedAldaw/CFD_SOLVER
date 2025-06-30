// ===== NUMERICAL SCHEMES IMPLEMENTATIONS =====

// === src/numerics/FluxScheme.cpp ===
#include "cfd/numerics/FluxScheme.hpp"
#include "cfd/physics/NavierStokes.hpp"
#include <cmath>
#include <algorithm>

namespace cfd::numerics {

// Helper functions
namespace {
    
// Roe average for compressible flow
void computeRoeAverage(const physics::CompressibleState& stateL,
                      const physics::CompressibleState& stateR,
                      Real& rho, Vector3& U, Real& H, Real& c) {
    // Extract primitive variables
    Real rhoL = stateL.density();
    Real rhoR = stateR.density();
    Vector3 UL = stateL.velocity();
    Vector3 UR = stateR.velocity();
    Real pL = stateL.pressure();
    Real pR = stateR.pressure();
    
    // Compute enthalpies
    Real HL = stateL.totalEnthalpy();
    Real HR = stateR.totalEnthalpy();
    
    // Roe averages
    Real sqrtRhoL = std::sqrt(rhoL);
    Real sqrtRhoR = std::sqrt(rhoR);
    Real sumSqrtRho = sqrtRhoL + sqrtRhoR;
    
    rho = sqrtRhoL * sqrtRhoR;
    U = (sqrtRhoL * UL + sqrtRhoR * UR) / sumSqrtRho;
    H = (sqrtRhoL * HL + sqrtRhoR * HR) / sumSqrtRho;
    
    // Sound speed
    Real gamma = 1.4; // Should come from physics model
    Real V2 = U.squaredNorm();
    c = std::sqrt((gamma - 1) * (H - 0.5 * V2));
}

// HLLC wave speeds
void computeHLLCWaveSpeeds(const physics::CompressibleState& stateL,
                          const physics::CompressibleState& stateR,
                          const Vector3& normal,
                          Real& SL, Real& SR, Real& SM) {
    Real rhoL = stateL.density();
    Real rhoR = stateR.density();
    Real unL = stateL.velocity().dot(normal);
    Real unR = stateR.velocity().dot(normal);
    Real pL = stateL.pressure();
    Real pR = stateR.pressure();
    Real cL = stateL.soundSpeed();
    Real cR = stateR.soundSpeed();
    
    // Pressure estimate (PVRS)
    Real pPVRS = 0.5 * (pL + pR) - 0.5 * (unR - unL) * 
                 0.5 * (rhoL + rhoR) * 0.5 * (cL + cR);
    pPVRS = std::max(Real(0), pPVRS);
    
    // Wave speed estimates
    Real qL = 1.0;
    if (pPVRS > pL) {
        Real gamma = 1.4;
        qL = std::sqrt(1 + (gamma + 1) / (2 * gamma) * (pPVRS / pL - 1));
    }
    
    Real qR = 1.0;
    if (pPVRS > pR) {
        Real gamma = 1.4;
        qR = std::sqrt(1 + (gamma + 1) / (2 * gamma) * (pPVRS / pR - 1));
    }
    
    SL = unL - cL * qL;
    SR = unR + cR * qR;
    
    // Contact wave speed
    Real num = pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR);
    Real den = rhoL * (SL - unL) - rhoR * (SR - unR);
    SM = num / (den + SMALL);
}

} // anonymous namespace

// Base flux scheme
template<typename State>
SharedPtr<FluxScheme<State>> FluxScheme<State>::create(FluxType type,
                                                       SharedPtr<Mesh> mesh) {
    switch (type) {
        case FluxType::ROE:
            return std::make_shared<RoeFlux<State>>(mesh);
        case FluxType::HLLC:
            return std::make_shared<HLLCFlux<State>>(mesh);
        case FluxType::AUSM:
            return std::make_shared<AUSMPlusFlux<State>>(mesh);
        case FluxType::CENTRAL:
            return std::make_shared<CentralFlux<State>>(mesh);
        default:
            throw std::runtime_error("Unknown flux type");
    }
}

// Roe flux implementation
template<>
VectorX RoeFlux<physics::CompressibleState>::compute(
    const physics::CompressibleState& stateL,
    const physics::CompressibleState& stateR,
    const Vector3& normal) const {
    
    VectorX flux(5);
    
    // Compute Roe averages
    Real rho, H, c;
    Vector3 U;
    computeRoeAverage(stateL, stateR, rho, U, H, c);
    
    // Normal velocity
    Real un = U.dot(normal);
    
    // Eigenvalues
    std::array<Real, 5> lambda = {un - c, un, un, un, un + c};
    
    // Compute flux difference
    VectorX deltaF = stateR.flux(normal) - stateL.flux(normal);
    VectorX deltaU = stateR.conserved - stateL.conserved;
    
    // Wave strengths (simplified - full implementation would compute eigenvectors)
    std::array<Real, 5> alpha;
    Real dp = stateR.pressure() - stateL.pressure();
    Real drho = stateR.density() - stateL.density();
    Real dun = stateR.velocity().dot(normal) - stateL.velocity().dot(normal);
    
    alpha[0] = (dp - rho * c * dun) / (2 * c * c);
    alpha[1] = drho - dp / (c * c);
    alpha[2] = 0; // Tangential velocity 1
    alpha[3] = 0; // Tangential velocity 2
    alpha[4] = (dp + rho * c * dun) / (2 * c * c);
    
    // Entropy fix
    const Real epsilon = 0.1 * c;
    for (auto& lam : lambda) {
        if (std::abs(lam) < epsilon) {
            lam = (lam * lam + epsilon * epsilon) / (2 * epsilon);
        }
    }
    
    // Average flux
    flux = 0.5 * (stateL.flux(normal) + stateR.flux(normal));
    
    // Add dissipation
    for (int i = 0; i < 5; ++i) {
        flux[i] -= 0.5 * std::abs(lambda[i]) * alpha[i];
    }
    
    return flux;
}

// HLLC flux implementation
template<>
VectorX HLLCFlux<physics::CompressibleState>::compute(
    const physics::CompressibleState& stateL,
    const physics::CompressibleState& stateR,
    const Vector3& normal) const {
    
    VectorX flux(5);
    
    // Compute wave speeds
    Real SL, SR, SM;
    computeHLLCWaveSpeeds(stateL, stateR, normal, SL, SR, SM);
    
    // Compute HLLC flux
    if (SL >= 0) {
        // Supersonic from left
        flux = stateL.flux(normal);
    } else if (SR <= 0) {
        // Supersonic from right
        flux = stateR.flux(normal);
    } else if (SM >= 0) {
        // Subsonic, sample from left star region
        const VectorX& FL = stateL.flux(normal);
        const VectorX& UL = stateL.conserved;
        Real unL = stateL.velocity().dot(normal);
        Real pL = stateL.pressure();
        
        // Star state pressure
        Real pStar = stateL.density() * (unL - SL) * (unL - SM) + pL;
        
        // HLLC flux
        Real factor = SM * (SL - unL) / (SL - SM);
        flux = FL + SL * (factor * UL - UL);
        
        // Pressure correction
        VectorX pCorr = VectorX::Zero(5);
        pCorr[1] = pStar * normal.x();
        pCorr[2] = pStar * normal.y();
        pCorr[3] = pStar * normal.z();
        pCorr[4] = pStar * SM;
        
        flux += factor * pCorr;
    } else {
        // Subsonic, sample from right star region
        const VectorX& FR = stateR.flux(normal);
        const VectorX& UR = stateR.conserved;
        Real unR = stateR.velocity().dot(normal);
        Real pR = stateR.pressure();
        
        // Star state pressure
        Real pStar = stateR.density() * (unR - SR) * (unR - SM) + pR;
        
        // HLLC flux
        Real factor = SM * (SR - unR) / (SR - SM);
        flux = FR + SR * (factor * UR - UR);
        
        // Pressure correction
        VectorX pCorr = VectorX::Zero(5);
        pCorr[1] = pStar * normal.x();
        pCorr[2] = pStar * normal.y();
        pCorr[3] = pStar * normal.z();
        pCorr[4] = pStar * SM;
        
        flux += factor * pCorr;
    }
    
    return flux;
}

// AUSM+ flux implementation
template<>
VectorX AUSMPlusFlux<physics::CompressibleState>::compute(
    const physics::CompressibleState& stateL,
    const physics::CompressibleState& stateR,
    const Vector3& normal) const {
    
    VectorX flux(5);
    
    // Extract variables
    Real rhoL = stateL.density();
    Real rhoR = stateR.density();
    Vector3 UL = stateL.velocity();
    Vector3 UR = stateR.velocity();
    Real pL = stateL.pressure();
    Real pR = stateR.pressure();
    Real cL = stateL.soundSpeed();
    Real cR = stateR.soundSpeed();
    
    // Interface sound speed
    Real c12 = 0.5 * (cL + cR);
    
    // Normal velocities
    Real unL = UL.dot(normal);
    Real unR = UR.dot(normal);
    
    // Mach numbers
    Real ML = unL / c12;
    Real MR = unR / c12;
    
    // Split Mach numbers (M+ and M-)
    Real MpL, MmR;
    if (std::abs(ML) <= 1) {
        MpL = 0.25 * (ML + 1) * (ML + 1);
    } else {
        MpL = 0.5 * (ML + std::abs(ML));
    }
    
    if (std::abs(MR) <= 1) {
        MmR = -0.25 * (MR - 1) * (MR - 1);
    } else {
        MmR = 0.5 * (MR - std::abs(MR));
    }
    
    // Interface Mach number
    Real M12 = MpL + MmR;
    
    // Split pressures
    Real alpha = 0.1875; // 3/16
    Real PpL, PmR;
    
    if (std::abs(ML) <= 1) {
        PpL = 0.25 * pL * (ML + 1) * (ML + 1) * (2 - ML) + 
              alpha * pL * ML * (ML * ML - 1) * (ML * ML - 1);
    } else {
        PpL = 0.5 * pL * (ML + std::abs(ML)) / ML;
    }
    
    if (std::abs(MR) <= 1) {
        PmR = 0.25 * pR * (MR - 1) * (MR - 1) * (2 + MR) - 
              alpha * pR * MR * (MR * MR - 1) * (MR * MR - 1);
    } else {
        PmR = 0.5 * pR * (MR - std::abs(MR)) / MR;
    }
    
    // Interface pressure
    Real p12 = PpL + PmR;
    
    // Mass flux
    Real mdot;
    if (M12 >= 0) {
        mdot = M12 * c12 * rhoL;
    } else {
        mdot = M12 * c12 * rhoR;
    }
    
    // Convective fluxes
    if (mdot >= 0) {
        flux[0] = mdot;
        flux[1] = mdot * UL.x() + p12 * normal.x();
        flux[2] = mdot * UL.y() + p12 * normal.y();
        flux[3] = mdot * UL.z() + p12 * normal.z();
        flux[4] = mdot * stateL.totalEnthalpy();
    } else {
        flux[0] = mdot;
        flux[1] = mdot * UR.x() + p12 * normal.x();
        flux[2] = mdot * UR.y() + p12 * normal.y();
        flux[3] = mdot * UR.z() + p12 * normal.z();
        flux[4] = mdot * stateR.totalEnthalpy();
    }
    
    return flux;
}

// Central flux (for testing/incompressible)
template<typename State>
VectorX CentralFlux<State>::compute(const State& stateL,
                                   const State& stateR,
                                   const Vector3& normal) const {
    // Simple average
    return 0.5 * (stateL.flux(normal) + stateR.flux(normal));
}

// Flux Jacobian computation
template<>
MatrixX RoeFlux<physics::CompressibleState>::jacobian(
    const physics::CompressibleState& state,
    const Vector3& normal) const {
    
    MatrixX A(5, 5);
    
    // Extract variables
    Real rho = state.density();
    Vector3 U = state.velocity();
    Real p = state.pressure();
    Real c = state.soundSpeed();
    Real gamma = 1.4; // Should come from physics
    
    Real un = U.dot(normal);
    Real V2 = U.squaredNorm();
    Real H = state.totalEnthalpy();
    
    // Flux Jacobian matrix
    A.setZero();
    
    // Row 0 (continuity)
    A(0, 0) = 0;
    A(0, 1) = normal.x();
    A(0, 2) = normal.y();
    A(0, 3) = normal.z();
    A(0, 4) = 0;
    
    // Row 1 (x-momentum)
    A(1, 0) = -U.x() * un + 0.5 * (gamma - 1) * V2 * normal.x();
    A(1, 1) = U.x() * normal.x() + un - (gamma - 1) * U.x() * normal.x();
    A(1, 2) = U.x() * normal.y() - (gamma - 1) * U.y() * normal.x();
    A(1, 3) = U.x() * normal.z() - (gamma - 1) * U.z() * normal.x();
    A(1, 4) = (gamma - 1) * normal.x();
    
    // Row 2 (y-momentum)
    A(2, 0) = -U.y() * un + 0.5 * (gamma - 1) * V2 * normal.y();
    A(2, 1) = U.y() * normal.x() - (gamma - 1) * U.x() * normal.y();
    A(2, 2) = U.y() * normal.y() + un - (gamma - 1) * U.y() * normal.y();
    A(2, 3) = U.y() * normal.z() - (gamma - 1) * U.z() * normal.y();
    A(2, 4) = (gamma - 1) * normal.y();
    
    // Row 3 (z-momentum)
    A(3, 0) = -U.z() * un + 0.5 * (gamma - 1) * V2 * normal.z();
    A(3, 1) = U.z() * normal.x() - (gamma - 1) * U.x() * normal.z();
    A(3, 2) = U.z() * normal.y() - (gamma - 1) * U.y() * normal.z();
    A(3, 3) = U.z() * normal.z() + un - (gamma - 1) * U.z() * normal.z();
    A(3, 4) = (gamma - 1) * normal.z();
    
    // Row 4 (energy)
    A(4, 0) = un * (0.5 * (gamma - 1) * V2 - H);
    A(4, 1) = H * normal.x() - (gamma - 1) * U.x() * un;
    A(4, 2) = H * normal.y() - (gamma - 1) * U.y() * un;
    A(4, 3) = H * normal.z() - (gamma - 1) * U.z() * un;
    A(4, 4) = gamma * un;
    
    return A;
}

// Explicit instantiations
template class FluxScheme<physics::CompressibleState>;
template class RoeFlux<physics::CompressibleState>;
template class HLLCFlux<physics::CompressibleState>;
template class AUSMPlusFlux<physics::CompressibleState>;
template class CentralFlux<physics::CompressibleState>;
