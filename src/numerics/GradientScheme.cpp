
// === src/numerics/GradientScheme.cpp ===
#include "cfd/numerics/GradientScheme.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/parallel/Communication.hpp"
#include <Eigen/Dense>

namespace cfd::numerics {

SharedPtr<GradientScheme> GradientScheme::create(GradientSchemeType type,
                                                 SharedPtr<Mesh> mesh) {
    switch (type) {
        case GradientSchemeType::GREEN_GAUSS:
            return std::make_shared<GreenGaussGradient>(mesh);
        case GradientSchemeType::LEAST_SQUARES:
            return std::make_shared<LeastSquaresGradient>(mesh);
        case GradientSchemeType::WEIGHTED_LEAST_SQUARES:
            return std::make_shared<WeightedLeastSquaresGradient>(mesh);
        default:
            throw std::runtime_error("Unknown gradient scheme type");
    }
}

// Green-Gauss gradient
VectorField GreenGaussGradient::gradient(const ScalarField& field) const {
    VectorField grad(mesh_, "grad(" + field.name() + ")");
    
    // Initialize to zero
    grad = Vector3::Zero();
    
    // First pass: compute face values
    std::vector<Real> faceValues(mesh_->numFaces());
    
    for (Index faceId = 0; faceId < mesh_->numFaces(); ++faceId) {
        const Face& face = mesh_->face(faceId);
        
        if (face.isBoundary()) {
            // Use boundary value
            faceValues[faceId] = field[face.owner()];
        } else {
            // Linear interpolation
            Real w = face.interpolationWeight();
            faceValues[faceId] = w * field[face.owner()] + 
                                (1 - w) * field[face.neighbor()];
        }
    }
    
    // Second pass: compute gradients
    for (Index cellId = 0; cellId < mesh_->numCells(); ++cellId) {
        const Cell& cell = mesh_->cell(cellId);
        Vector3& cellGrad = grad[cellId];
        
        for (const Face* face : cell.faces()) {
            Real faceValue = faceValues[face->id()];
            Vector3 Sf = face->normal() * face->area();
            
            // Correct sign for neighbor cells
            if (face->owner() != cellId) {
                Sf = -Sf;
            }
            
            cellGrad += faceValue * Sf;
        }
        
        cellGrad /= cell.volume();
    }
    
    // Update ghost cells
    grad.updateBoundaryConditions();
    
    return grad;
}

TensorField GreenGaussGradient::gradient(const VectorField& field) const {
    TensorField grad(mesh_, "grad(" + field.name() + ")");
    
    // Compute gradient of each component
    for (int comp = 0; comp < 3; ++comp) {
        ScalarField component = field.component(comp);
        VectorField gradComp = gradient(component);
        
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            grad[i].row(comp) = gradComp[i];
        }
    }
    
    return grad;
}

// Least squares gradient
void LeastSquaresGradient::computeStencils() {
    if (stencilsComputed_) return;
    
    cellStencils_.resize(mesh_->numCells());
    lsqMatrices_.resize(mesh_->numCells());
    
    for (Index cellId = 0; cellId < mesh_->numCells(); ++cellId) {
        const Cell& cell = mesh_->cell(cellId);
        const Vector3& xc = cell.center();
        
        // Build stencil from neighbors
        std::vector<Index>& stencil = cellStencils_[cellId];
        
        // Add face neighbors
        for (const Face* face : cell.faces()) {
            if (!face->isBoundary()) {
                Index neighborId = (face->owner() == cellId) ? 
                                  face->neighbor() : face->owner();
                stencil.push_back(neighborId);
            }
        }
        
        // Extended stencil for better accuracy (optional)
        if (useExtendedStencil_) {
            std::set<Index> extendedStencil(stencil.begin(), stencil.end());
            
            for (Index neighborId : stencil) {
                const Cell& neighbor = mesh_->cell(neighborId);
                for (const Face* face : neighbor.faces()) {
                    if (!face->isBoundary()) {
                        Index secondNeighbor = (face->owner() == neighborId) ?
                                             face->neighbor() : face->owner();
                        extendedStencil.insert(secondNeighbor);
                    }
                }
            }
            
            extendedStencil.erase(cellId); // Remove self
            stencil.assign(extendedStencil.begin(), extendedStencil.end());
        }
        
        // Build least squares matrix
        const int n = stencil.size();
        MatrixX A(n, 3);
        
        for (int i = 0; i < n; ++i) {
            const Vector3& xn = mesh_->cell(stencil[i]).center();
            Vector3 dx = xn - xc;
            A.row(i) = dx;
        }
        
        // Compute pseudo-inverse: (A^T A)^{-1} A^T
        MatrixX AtA = A.transpose() * A;
        
        // Add regularization for stability
        AtA.diagonal().array() += 1e-10;
        
        lsqMatrices_[cellId] = AtA.inverse() * A.transpose();
    }
    
    stencilsComputed_ = true;
}

VectorField LeastSquaresGradient::gradient(const ScalarField& field) const {
    const_cast<LeastSquaresGradient*>(this)->computeStencils();
    
    VectorField grad(mesh_, "grad(" + field.name() + ")");
    
    for (Index cellId = 0; cellId < mesh_->numCells(); ++cellId) {
        const std::vector<Index>& stencil = cellStencils_[cellId];
        const MatrixX& lsqMatrix = lsqMatrices_[cellId];
        
        // Build RHS vector
        VectorX b(stencil.size());
        for (size_t i = 0; i < stencil.size(); ++i) {
            b[i] = field[stencil[i]] - field[cellId];
        }
        
        // Compute gradient
        grad[cellId] = lsqMatrix * b;
    }
    
    // Apply gradient limiters if needed
    if (useLimiters_) {
        applyLimiters(field, grad);
    }
    
    grad.updateBoundaryConditions();
    
    return grad;
}

void LeastSquaresGradient::applyLimiters(const ScalarField& field,
                                         VectorField& grad) const {
    // Barth-Jespersen limiter
    std::vector<Real> limiters(mesh_->numCells(), 1.0);
    
    for (Index cellId = 0; cellId < mesh_->numCells(); ++cellId) {
        const Cell& cell = mesh_->cell(cellId);
        Real phiC = field[cellId];
        const Vector3& gradC = grad[cellId];
        
        Real phiMin = phiC;
        Real phiMax = phiC;
        
        // Find min/max in neighborhood
        for (Index neighborId : cellStencils_[cellId]) {
            phiMin = std::min(phiMin, field[neighborId]);
            phiMax = std::max(phiMax, field[neighborId]);
        }
        
        // Compute limiter
        Real psi = 1.0;
        
        for (const Face* face : cell.faces()) {
            Vector3 dx = face->center() - cell.center();
            Real phiF = phiC + gradC.dot(dx);
            
            if (phiF > phiMax) {
                psi = std::min(psi, (phiMax - phiC) / (phiF - phiC + SMALL));
            } else if (phiF < phiMin) {
                psi = std::min(psi, (phiMin - phiC) / (phiF - phiC + SMALL));
            }
        }
        
        limiters[cellId] = std::max(Real(0), psi);
    }
    
    // Apply limiters
    for (Index cellId = 0; cellId < mesh_->numCells(); ++cellId) {
        grad[cellId] *= limiters[cellId];
    }
}

// Weighted least squares gradient
VectorField WeightedLeastSquaresGradient::gradient(const ScalarField& field) const {
    const_cast<WeightedLeastSquaresGradient*>(this)->computeStencils();
    
    VectorField grad(mesh_, "grad(" + field.name() + ")");
    
    for (Index cellId = 0; cellId < mesh_->numCells(); ++cellId) {
        const Cell& cell = mesh_->cell(cellId);
        const Vector3& xc = cell.center();
        const std::vector<Index>& stencil = cellStencils_[cellId];
        
        // Build weighted least squares system
        Matrix3 AtWA = Matrix3::Zero();
        Vector3 AtWb = Vector3::Zero();
        
        for (Index neighborId : stencil) {
            const Vector3& xn = mesh_->cell(neighborId).center();
            Vector3 dx = xn - xc;
            Real dist = dx.norm();
            
            // Inverse distance weighting
            Real w = 1.0 / (dist + SMALL);
            
            // Add to normal equations
            AtWA += w * dx * dx.transpose();
            AtWb += w * dx * (field[neighborId] - field[cellId]);
        }
        
        // Solve for gradient
        grad[cellId] = AtWA.ldlt().solve(AtWb);
    }
    
    if (useLimiters_) {
        applyLimiters(field, grad);
    }
    
    grad.updateBoundaryConditions();
    
    return grad;
}
