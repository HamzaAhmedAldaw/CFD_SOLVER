// === src/numerics/Limiter.cpp ===
#include "cfd/numerics/Limiter.hpp"
#include <algorithm>
#include <cmath>

namespace cfd::numerics {

SharedPtr<Limiter> Limiter::create(LimiterType type, SharedPtr<Mesh> mesh) {
    switch (type) {
        case LimiterType::NONE:
            return std::make_shared<NoLimiter>(mesh);
        case LimiterType::BARTH_JESPERSEN:
            return std::make_shared<BarthJespersenLimiter>(mesh);
        case LimiterType::VENKATAKRISHNAN:
            return std::make_shared<VenkatakrishnanLimiter>(mesh);
        case LimiterType::MINMOD:
            return std::make_shared<MinmodLimiter>(mesh);
        case LimiterType::VANLEER:
            return std::make_shared<VanLeerLimiter>(mesh);
        case LimiterType::SUPERBEE:
            return std::make_shared<SuperbeeLimiter>(mesh);
        default:
            throw std::runtime_error("Unknown limiter type");
    }
}

// Barth-Jespersen limiter
Vector3 BarthJespersenLimiter::limit(const Vector3& gradient,
                                    const Vector3& dr,
                                    const ScalarField& field,
                                    Index cellId) const {
    const Cell& cell = mesh_->cell(cellId);
    Real phiC = field[cellId];
    
    // Find min/max in neighborhood
    Real phiMin = phiC;
    Real phiMax = phiC;
    
    for (const Face* face : cell.faces()) {
        if (!face->isBoundary()) {
            Index neighborId = (face->owner() == cellId) ? 
                              face->neighbor() : face->owner();
            phiMin = std::min(phiMin, field[neighborId]);
            phiMax = std::max(phiMax, field[neighborId]);
        }
    }
    
    // Compute limiter
    Real psi = 1.0;
    Real phiF = phiC + gradient.dot(dr);
    
    if (phiF > phiMax) {
        psi = (phiMax - phiC) / (phiF - phiC + SMALL);
    } else if (phiF < phiMin) {
        psi = (phiMin - phiC) / (phiF - phiC + SMALL);
    }
    
    psi = std::max(Real(0), std::min(Real(1), psi));
    
    return psi * gradient;
}

// Venkatakrishnan limiter
void VenkatakrishnanLimiter::computeLimiterConstants() {
    if (constantsComputed_) return;
    
    lengthScale_.resize(mesh_->numCells());
    
    for (Index cellId = 0; cellId < mesh_->numCells(); ++cellId) {
        const Cell& cell = mesh_->cell(cellId);
        
        // Characteristic length scale (cube root of volume)
        lengthScale_[cellId] = std::pow(cell.volume(), 1.0/3.0);
    }
    
    constantsComputed_ = true;
}

Vector3 VenkatakrishnanLimiter::limit(const Vector3& gradient,
                                     const Vector3& dr,
                                     const ScalarField& field,
                                     Index cellId) const {
    const_cast<VenkatakrishnanLimiter*>(this)->computeLimiterConstants();
    
    const Cell& cell = mesh_->cell(cellId);
    Real phiC = field[cellId];
    Real h = lengthScale_[cellId];
    
    // Venkat's constant
    Real eps2 = K_ * K_ * K_ * h * h * h;
    
    // Find min/max in neighborhood
    Real phiMin = phiC;
    Real phiMax = phiC;
    
    for (const Face* face : cell.faces()) {
        if (!face->isBoundary()) {
            Index neighborId = (face->owner() == cellId) ? 
                              face->neighbor() : face->owner();
            phiMin = std::min(phiMin, field[neighborId]);
            phiMax = std::max(phiMax, field[neighborId]);
        }
    }
    
    // Compute limiter
    Real phiF = phiC + gradient.dot(dr);
    Real deltaPlus = phiMax - phiC;
    Real deltaMinus = phiC - phiMin;
    Real delta = phiF - phiC;
    
    Real psi;
    if (delta > 0) {
        Real denom = delta * delta + eps2;
        psi = (deltaPlus * deltaPlus + eps2 + 2 * delta * deltaPlus) / denom;
    } else if (delta < 0) {
        Real denom = delta * delta + eps2;
        psi = (deltaMinus * deltaMinus + eps2 - 2 * delta * deltaMinus) / denom;
    } else {
        psi = 1.0;
    }
    
    psi = std::min(Real(1), psi);
    
    return psi * gradient;
}

// Minmod limiter
Real MinmodLimiter::minmod(Real a, Real b) {
    if (a * b <= 0) return 0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

Real MinmodLimiter::minmod(Real a, Real b, Real c) {
    return minmod(a, minmod(b, c));
}

Vector3 MinmodLimiter::limit(const Vector3& gradient,
                            const Vector3& dr,
                            const ScalarField& field,
                            Index cellId) const {
    const Cell& cell = mesh_->cell(cellId);
    Real phiC = field[cellId];
    
    // Compute forward and backward differences
    Vector3 limitedGrad = gradient;
    
    for (int comp = 0; comp < 3; ++comp) {
        Real grad_comp = gradient[comp];
        
        // Find min/max slopes
        Real slopeMin = grad_comp;
        Real slopeMax = grad_comp;
        
        for (const Face* face : cell.faces()) {
            if (!face->isBoundary()) {
                Index neighborId = (face->owner() == cellId) ? 
                                  face->neighbor() : face->owner();
                Vector3 dx = mesh_->cell(neighborId).center() - cell.center();
                Real slope = (field[neighborId] - phiC) / dx[comp];
                
                slopeMin = std::min(slopeMin, slope);
                slopeMax = std::max(slopeMax, slope);
            }
        }
        
        limitedGrad[comp] = minmod(grad_comp, 2*slopeMin, 2*slopeMax);
    }
    
    return limitedGrad;
}

// Van Leer limiter
Vector3 VanLeerLimiter::limit(const Vector3& gradient,
                             const Vector3& dr,
                             const ScalarField& field,
                             Index cellId) const {
    const Cell& cell = mesh_->cell(cellId);
    Real phiC = field[cellId];
    Real phiF = phiC + gradient.dot(dr);
    
    // Find upwind and downwind values
    Real phiU = phiC;
    Real phiD = phiF;
    
    // Find far upwind value (simplified)
    Real phiUU = phiC;
    for (const Face* face : cell.faces()) {
        if (!face->isBoundary()) {
            Index neighborId = (face->owner() == cellId) ? 
                              face->neighbor() : face->owner();
            Vector3 dx = mesh_->cell(neighborId).center() - cell.center();
            if (dx.dot(dr) < 0) { // Upwind direction
                phiUU = field[neighborId];
                break;
            }
        }
    }
    
    // Compute gradient ratio
    Real r = (phiU - phiUU) / (phiD - phiU + SMALL);
    
    // Van Leer limiter function
    Real psi = (r + std::abs(r)) / (1 + std::abs(r));
    
    return psi * gradient;
}
