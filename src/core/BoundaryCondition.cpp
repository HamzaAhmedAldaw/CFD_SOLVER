// === src/core/BoundaryCondition.cpp ===
#include "cfd/core/BoundaryCondition.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/core/Mesh.hpp"
#include "cfd/io/Logger.hpp"

namespace cfd {

// Base boundary condition
template<typename T>
BoundaryCondition<T>::BoundaryCondition(const std::string& patchName, BCType type)
    : patchName_(patchName), type_(type) {
}

// Dirichlet BC
template<typename T>
DirichletBC<T>::DirichletBC(const std::string& patchName, const T& value)
    : BoundaryCondition<T>(patchName, BCType::DIRICHLET), value_(value) {
}

template<typename T>
DirichletBC<T>::DirichletBC(const std::string& patchName,
                           const std::function<T(const Vector3&, Real)>& func)
    : BoundaryCondition<T>(patchName, BCType::DIRICHLET), function_(func) {
}

template<>
void DirichletBC<Real>::apply(Field<Real>& field) const {
    const Mesh& mesh = *field.mesh();
    const BoundaryPatch& patch = mesh.boundaryPatch(this->patchName_);
    
    for (const Face* face : patch.faces()) {
        Index cellId = face->owner();
        
        if (function_) {
            field[cellId] = function_(face->center(), 0.0); // Time = 0 for now
        } else {
            field[cellId] = value_;
        }
    }
}

template<>
void DirichletBC<Vector3>::apply(Field<Vector3>& field) const {
    const Mesh& mesh = *field.mesh();
    const BoundaryPatch& patch = mesh.boundaryPatch(this->patchName_);
    
    for (const Face* face : patch.faces()) {
        Index cellId = face->owner();
        
        if (function_) {
            field[cellId] = function_(face->center(), 0.0);
        } else {
            field[cellId] = value_;
        }
    }
}

// Neumann BC
template<typename T>
NeumannBC<T>::NeumannBC(const std::string& patchName, const T& gradient)
    : BoundaryCondition<T>(patchName, BCType::NEUMANN), gradient_(gradient) {
}

template<>
void NeumannBC<Real>::apply(Field<Real>& field) const {
    const Mesh& mesh = *field.mesh();
    const BoundaryPatch& patch = mesh.boundaryPatch(this->patchName_);
    
    for (const Face* face : patch.faces()) {
        Index cellId = face->owner();
        const Cell& cell = mesh.cell(cellId);
        
        // Distance from cell center to face center
        Vector3 d = face->center() - cell.center();
        Real dist = d.norm();
        
        // Apply gradient: phi_f = phi_c + grad * d
        // For zero gradient, phi_f = phi_c
        field[cellId] = field[cellId]; // No change for zero gradient
        
        if (gradient_ != 0.0) {
            // This would need modification to handle face values
            // For now, simplified implementation
        }
    }
}

// Robin BC
template<typename T>
RobinBC<T>::RobinBC(const std::string& patchName, Real alpha, Real beta, const T& gamma)
    : BoundaryCondition<T>(patchName, BCType::ROBIN),
      alpha_(alpha), beta_(beta), gamma_(gamma) {
}

template<>
void RobinBC<Real>::apply(Field<Real>& field) const {
    const Mesh& mesh = *field.mesh();
    const BoundaryPatch& patch = mesh.boundaryPatch(this->patchName_);
    
    for (const Face* face : patch.faces()) {
        Index cellId = face->owner();
        
        // Robin BC: alpha * phi + beta * dphi/dn = gamma
        // Simplified implementation
        if (std::abs(alpha_) > EPSILON) {
            field[cellId] = (gamma_ - beta_ * 0.0) / alpha_; // Assume zero gradient
        }
    }
}

// No-slip wall BC for velocity
NoSlipWallBC::NoSlipWallBC(const std::string& patchName)
    : BoundaryCondition<Vector3>(patchName, BCType::WALL) {
}

void NoSlipWallBC::apply(Field<Vector3>& field) const {
    const Mesh& mesh = *field.mesh();
    const BoundaryPatch& patch = mesh.boundaryPatch(patchName_);
    
    for (const Face* face : patch.faces()) {
        Index cellId = face->owner();
        field[cellId] = Vector3::Zero();
    }
}

// Moving wall BC
MovingWallBC::MovingWallBC(const std::string& patchName, const Vector3& velocity)
    : BoundaryCondition<Vector3>(patchName, BCType::WALL), velocity_(velocity) {
}

void MovingWallBC::apply(Field<Vector3>& field) const {
    const Mesh& mesh = *field.mesh();
    const BoundaryPatch& patch = mesh.boundaryPatch(patchName_);
    
    for (const Face* face : patch.faces()) {
        Index cellId = face->owner();
        field[cellId] = velocity_;
    }
}

// Inlet BC
InletBC::InletBC(const std::string& patchName)
    : BoundaryCondition<Vector3>(patchName, BCType::INLET) {
}

void InletBC::setVelocity(const Vector3& velocity) {
    velocity_ = velocity;
}

void InletBC::setVelocityProfile(const std::function<Vector3(const Vector3&)>& profile) {
    velocityProfile_ = profile;
}

void InletBC::apply(Field<Vector3>& field) const {
    const Mesh& mesh = *field.mesh();
    const BoundaryPatch& patch = mesh.boundaryPatch(patchName_);
    
    for (const Face* face : patch.faces()) {
        Index cellId = face->owner();
        
        if (velocityProfile_) {
            field[cellId] = velocityProfile_(face->center());
        } else {
            field[cellId] = velocity_;
        }
    }
}

// Outlet BC
OutletBC::OutletBC(const std::string& patchName)
    : BoundaryCondition<Real>(patchName, BCType::OUTLET), pressure_(0.0) {
}

void OutletBC::setPressure(Real pressure) {
    pressure_ = pressure;
}

void OutletBC::apply(Field<Real>& field) const {
    const Mesh& mesh = *field.mesh();
    const BoundaryPatch& patch = mesh.boundaryPatch(patchName_);
    
    for (const Face* face : patch.faces()) {
        Index cellId = face->owner();
        field[cellId] = pressure_;
    }
}

// Symmetry BC
template<typename T>
SymmetryBC<T>::SymmetryBC(const std::string& patchName)
    : BoundaryCondition<T>(patchName, BCType::SYMMETRY) {
}

template<>
void SymmetryBC<Real>::apply(Field<Real>& field) const {
    // For scalar fields, symmetry means zero gradient
    // No modification needed
}

template<>
void SymmetryBC<Vector3>::apply(Field<Vector3>& field) const {
    const Mesh& mesh = *field.mesh();
    const BoundaryPatch& patch = mesh.boundaryPatch(this->patchName_);
    
    for (const Face* face : patch.faces()) {
        Index cellId = face->owner();
        const Vector3& n = face->normal();
        
        // Remove normal component: U = U - (U·n)n
        Vector3& U = field[cellId];
        U -= U.dot(n) * n;
    }
}

// Periodic BC
template<typename T>
PeriodicBC<T>::PeriodicBC(const std::string& patchName, const std::string& neighborPatch)
    : BoundaryCondition<T>(patchName, BCType::PERIODIC), neighborPatch_(neighborPatch) {
}

template<typename T>
void PeriodicBC<T>::apply(Field<T>& field) const {
    // Periodic BC requires special handling during matrix assembly
    // This is a placeholder implementation
    io::Logger::instance().warn("Periodic BC not fully implemented");
}

// Explicit instantiations
template class BoundaryCondition<Real>;
template class BoundaryCondition<Vector3>;
template class DirichletBC<Real>;
template class DirichletBC<Vector3>;
template class NeumannBC<Real>;
template class NeumannBC<Vector3>;
template class RobinBC<Real>;
template class RobinBC<Vector3>;
template class SymmetryBC<Real>;
template class SymmetryBC<Vector3>;
template class PeriodicBC<Real>;
template class PeriodicBC<Vector3>;

} // namespace cfd
