// === src/numerics/Interpolation.cpp ===
#include "cfd/numerics/Interpolation.hpp"
#include <algorithm>

namespace cfd::numerics {

// Linear interpolation
Real LinearInterpolation::interpolate(const ScalarField& field,
                                     const Face& face) const {
    if (face.isBoundary()) {
        return field[face.owner()];
    }
    
    Real w = face.interpolationWeight();
    return w * field[face.owner()] + (1 - w) * field[face.neighbor()];
}

Vector3 LinearInterpolation::interpolate(const VectorField& field,
                                        const Face& face) const {
    if (face.isBoundary()) {
        return field[face.owner()];
    }
    
    Real w = face.interpolationWeight();
    return w * field[face.owner()] + (1 - w) * field[face.neighbor()];
}

// Upwind interpolation
Real UpwindInterpolation::interpolate(const ScalarField& field,
                                     const Face& face,
                                     const VectorField& velocity) const {
    Real phi = velocity[face.owner()].dot(face.normal());
    
    if (face.isBoundary()) {
        return field[face.owner()];
    }
    
    // Upwind based on flux direction
    if (phi >= 0) {
        return field[face.owner()];
    } else {
        return field[face.neighbor()];
    }
}

// MUSCL interpolation with limiters
void MUSCLInterpolation::reconstruct(const ScalarField& field,
                                    const VectorField& gradient,
                                    const Face& face,
                                    Real& phiL, Real& phiR) const {
    Index owner = face.owner();
    const Cell& ownerCell = mesh_->cell(owner);
    
    // Left state reconstruction
    Vector3 dL = face.center() - ownerCell.center();
    phiL = field[owner] + limiter_->limit(gradient[owner], dL, field, owner).dot(dL);
    
    if (!face.isBoundary()) {
        Index neighbor = face.neighbor();
        const Cell& neighborCell = mesh_->cell(neighbor);
        
        // Right state reconstruction
        Vector3 dR = face.center() - neighborCell.center();
        phiR = field[neighbor] + limiter_->limit(gradient[neighbor], dR, field, neighbor).dot(dR);
    } else {
        // Boundary face
        phiR = phiL; // or apply boundary condition
    }
}

// High-order interpolation (WENO5)
Real HighOrderInterpolation::interpolate(const ScalarField& field,
                                        const Face& face,
                                        InterpolationOrder order) const {
    if (order == InterpolationOrder::FIRST) {
        return LinearInterpolation(mesh_).interpolate(field, face);
    }
    
    if (face.isBoundary()) {
        return field[face.owner()];
    }
    
    // WENO5 requires extended stencil
    // Simplified implementation - full WENO5 would be more complex
    
    Index owner = face.owner();
    Index neighbor = face.neighbor();
    
    // Find extended stencil
    std::vector<Index> stencil;
    stencil.push_back(owner);
    stencil.push_back(neighbor);
    
    // Add neighbors of owner and neighbor
    for (const Face* f : mesh_->cell(owner).faces()) {
        if (!f->isBoundary() && f->neighbor() != neighbor) {
            stencil.push_back(f->neighbor());
        }
    }
    
    for (const Face* f : mesh_->cell(neighbor).faces()) {
        if (!f->isBoundary() && f->neighbor() != owner) {
            stencil.push_back(f->neighbor());
        }
    }
    
    // Simple high-order interpolation (not full WENO)
    if (stencil.size() >= 4) {
        // Cubic interpolation
        // Implementation simplified
        return 0.5 * (field[owner] + field[neighbor]);
    } else {
        // Fall back to linear
        Real w = face.interpolationWeight();
        return w * field[owner] + (1 - w) * field[neighbor];
    }
}

// TVD interpolation
Real TVDInterpolation::interpolate(const ScalarField& field,
                                  const Face& face,
                                  const VectorField& velocity) const {
    if (face.isBoundary()) {
        return field[face.owner()];
    }
    
    Real phi = velocity[face.owner()].dot(face.normal());
    Index upwind, downwind;
    
    if (phi >= 0) {
        upwind = face.owner();
        downwind = face.neighbor();
    } else {
        upwind = face.neighbor();
        downwind = face.owner();
    }
    
    // Find far upwind value
    Real phiU = field[upwind];
    Real phiD = field[downwind];
    Real phiUU = phiU; // Need to find actual far upwind value
    
    // Compute gradient ratio
    Real r = (phiU - phiUU) / (phiD - phiU + SMALL);
    
    // Apply TVD limiter
    Real psi = tvdLimiter(r, limiterType_);
    
    // TVD interpolation
    return phiU + 0.5 * psi * (phiD - phiU);
}

Real TVDInterpolation::tvdLimiter(Real r, TVDLimiterType type) const {
    switch (type) {
        case TVDLimiterType::MINMOD:
            return std::max(Real(0), std::min(Real(1), r));
            
        case TVDLimiterType::VANLEER:
            return (r + std::abs(r)) / (1 + std::abs(r));
            
        case TVDLimiterType::SUPERBEE:
            return std::max(Real(0), std::max(std::min(Real(2)*r, Real(1)), 
                                             std::min(r, Real(2))));
            
        case TVDLimiterType::MC:
            return std::max(Real(0), std::min(std::min((1+r)/2, Real(2)), 
                                             Real(2)*r));
            
        default:
            return 0;
    }
}
