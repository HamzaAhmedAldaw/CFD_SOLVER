// === src/solvers/PressureVelocityCoupling.cpp ===
#include "cfd/solvers/PressureVelocityCoupling.hpp"
#include "cfd/numerics/GradientScheme.hpp"
#include "cfd/numerics/Interpolation.hpp"

namespace cfd::solvers {

PressureVelocityCoupling::PressureVelocityCoupling(
    SharedPtr<Mesh> mesh,
    const PVCouplingSettings& settings)
    : mesh_(mesh), settings_(settings) {
    
    // Create linear solver for pressure equation
    LinearSolver::SolverSettings linearSettings;
    linearSettings.tolerance = settings.pressureTolerance;
    linearSettings.maxIterations = settings.pressureMaxIterations;
    
    pressureSolver_ = LinearSolver::create(
        LinearSolverType::CG, linearSettings);
    
    // Create preconditioner
    auto precond = Preconditioner::create(PreconditionerType::ILU0);
    pressureSolver_->setPreconditioner(precond);
    
    // Allocate face flux field
    phi_ = std::make_shared<ScalarField>(mesh, "phi", FieldLocation::FACE);
}

void PressureVelocityCoupling::solvePressureCorrection(
    VectorField& U,
    ScalarField& p,
    ScalarField& pCorr) {
    
    // Compute face fluxes from velocity
    computeFaceFluxes(U);
    
    // Assemble pressure correction equation
    SparseMatrix Ap;
    VectorX bp;
    assemblePressureEquation(U, Ap, bp);
    
    // Solve for pressure correction
    VectorX xp(mesh_->numCells());
    xp.setZero();
    
    bool converged = pressureSolver_->solve(Ap, bp, xp);
    
    if (!converged) {
        logger_->warn("Pressure correction did not converge");
    }
    
    // Unpack solution
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        pCorr[i] = xp[i];
    }
    
    // Apply under-relaxation to pressure
    Real alphaP = settings_.pressureRelaxation;
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        p[i] += alphaP * pCorr[i];
    }
}

void PressureVelocityCoupling::correctVelocity(VectorField& U,
                                              const ScalarField& pCorr) {
    // Correct cell velocities
    numerics::GreenGaussGradient gradScheme(mesh_);
    VectorField gradPCorr = gradScheme.gradient(pCorr);
    
    for (Index cellId = 0; cellId < mesh_->numCells(); ++cellId) {
        const Cell& cell = mesh_->cell(cellId);
        
        // Velocity correction: U' = -1/ap * grad(p')
        // where ap is the diagonal coefficient from momentum equation
        Real ap = 1.0; // Simplified - should come from momentum equation
        
        U[cellId] -= gradPCorr[cellId] / ap;
    }
    
    // Correct face fluxes
    correctFaceFluxes(pCorr);
}

void PressureVelocityCoupling::computeFaceFluxes(const VectorField& U) {
    for (Index faceId = 0; faceId < mesh_->numFaces(); ++faceId) {
        const Face& face = mesh_->face(faceId);
        
        if (face.isBoundary()) {
            // Boundary face flux
            const Vector3& Ub = U[face.owner()];
            (*phi_)[faceId] = Ub.dot(face.normal()) * face.area();
        } else {
            // Internal face - Rhie-Chow interpolation
            Real phiRC = rhieChowInterpolation(U, face);
            (*phi_)[faceId] = phiRC * face.area();
        }
    }
}

Real PressureVelocityCoupling::rhieChowInterpolation(
    const VectorField& U,
    const Face& face) const {
    
    Index owner = face.owner();
    Index neighbor = face.neighbor();
    
    // Linear interpolation of velocity
    Real w = face.interpolationWeight();
    Vector3 Uf = w * U[owner] + (1 - w) * U[neighbor];
    
    // Rhie-Chow correction
    // phi_f = (U_f)_bar · n + d_f * (p_N - p_P)
    // where d_f is related to 1/ap
    
    // Simplified implementation
    Real phiLinear = Uf.dot(face.normal());
    
    // Pressure gradient contribution (would need pressure field)
    Real dpdn = 0.0; // Placeholder
    Real df = 1.0;   // Should be computed from momentum equation
    
    return phiLinear + df * dpdn;
}

void PressureVelocityCoupling::assemblePressureEquation(
    const VectorField& U,
    SparseMatrix& Ap,
    VectorX& bp) {
    
    const Index n = mesh_->numCells();
    Ap.resize(n, n);
    bp.resize(n);
    bp.setZero();
    
    std::vector<Triplet> triplets;
    triplets.reserve(n * 7); // Estimate
    
    // For each cell
    for (Index cellId = 0; cellId < n; ++cellId) {
        const Cell& cell = mesh_->cell(cellId);
        Real diagCoeff = 0.0;
        
        // For each face
        for (const Face* face : cell.faces()) {
            Index faceId = face->id();
            
            if (face->isBoundary()) {
                // Boundary contribution
                handlePressureBoundary(*face, cellId, diagCoeff, bp[cellId]);
            } else {
                // Internal face
                Index neighborId = (face->owner() == cellId) ? 
                                  face->neighbor() : face->owner();
                
                // Geometric factor
                Real gf = computeGeometricFactor(*face, cellId);
                
                // Add to matrix
                triplets.emplace_back(cellId, neighborId, -gf);
                diagCoeff += gf;
                
                // Add flux imbalance to RHS
                Real sign = (face->owner() == cellId) ? 1.0 : -1.0;
                bp[cellId] -= sign * (*phi_)[faceId];
            }
        }
        
        // Diagonal entry
        triplets.emplace_back(cellId, cellId, diagCoeff);
    }
    
    Ap.setFromTriplets(triplets.begin(), triplets.end());
}

Real PressureVelocityCoupling::computeGeometricFactor(
    const Face& face,
    Index cellId) const {
    
    Index owner = face.owner();
    Index neighbor = face.neighbor();
    
    const Cell& ownerCell = mesh_->cell(owner);
    const Cell& neighborCell = mesh_->cell(neighbor);
    
    // Distance between cell centers
    Vector3 d = neighborCell.center() - ownerCell.center();
    Real dist = d.norm();
    
    // Face area and normal
    Real Af = face.area();
    const Vector3& n = face.normal();
    
    // Non-orthogonal correction factor
    Real cosTheta = std::abs(d.normalized().dot(n));
    cosTheta = std::max(cosTheta, Real(0.1)); // Limit for stability
    
    // Geometric factor: Af * |n·d| / |d|²
    return Af * cosTheta / dist;
}

void PressureVelocityCoupling::handlePressureBoundary(
    const Face& face,
    Index cellId,
    Real& diagCoeff,
    Real& source) {
    
    // Simplified boundary handling
    // Full implementation would handle different BC types
    
    // Zero gradient (most common for pressure)
    // No contribution to matrix
    
    // Fixed value would contribute to source term
}

void PressureVelocityCoupling::correctFaceFluxes(const ScalarField& pCorr) {
    for (Index faceId = 0; faceId < mesh_->numFaces(); ++faceId) {
        const Face& face = mesh_->face(faceId);
        
        if (!face.isBoundary()) {
            // Flux correction: phi' = -d_f * (?p')_f · n * A_f
            Index owner = face.owner();
            Index neighbor = face.neighbor();
            
            Real dpdn = (pCorr[neighbor] - pCorr[owner]) / 
                       (mesh_->cell(neighbor).center() - 
                        mesh_->cell(owner).center()).norm();
            
            Real df = 1.0; // Should be from momentum equation
            
            (*phi_)[faceId] -= df * dpdn * face.area();
        }
    }
}
