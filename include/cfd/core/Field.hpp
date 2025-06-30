#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Mesh.hpp"
#include "cfd/core/BoundaryCondition.hpp"
#include <string>
#include <memory>

namespace cfd {

template<typename T>
class Field {
public:
    using value_type = T;
    using boundary_map = std::unordered_map<std::string, SharedPtr<BoundaryCondition<T>>>;
    
    // Constructors
    Field(const Mesh& mesh, const std::string& name, FieldType type);
    Field(const Mesh& mesh, const std::string& name, const T& initialValue);
    
    // Copy and move semantics
    Field(const Field& other);
    Field(Field&& other) noexcept;
    Field& operator=(const Field& other);
    Field& operator=(Field&& other) noexcept;
    
    // Field information
    const std::string& name() const { return name_; }
    FieldType fieldType() const { return type_; }
    const Mesh& mesh() const { return *mesh_; }
    
    // Data access
    T& operator[](Index i) { return data_[i]; }
    const T& operator[](Index i) const { return data_[i]; }
    
    T& cell(Index i) { return data_[i]; }
    const T& cell(Index i) const { return data_[i]; }
    
    T& face(Index i) { return faceData_[i]; }
    const T& face(Index i) const { return faceData_[i]; }
    
    // Boundary conditions
    void setBoundaryCondition(const std::string& patchName, 
                             SharedPtr<BoundaryCondition<T>> bc);
    BoundaryCondition<T>& boundaryCondition(const std::string& patchName);
    const BoundaryCondition<T>& boundaryCondition(const std::string& patchName) const;
    
    // Field operations
    void initialize(const T& value);
    void initialize(std::function<T(const Vector3&)> func);
    
    // Interpolation
    void interpolateToFaces();
    T interpolate(const Vector3& position) const;
    
    // Gradient computation
    Field<typename GradientType<T>::type> gradient() const;
    
    // Time management
    void storeOldTime();
    void storeOldOldTime();
    const Field& oldTime() const { return *oldTime_; }
    const Field& oldOldTime() const { return *oldOldTime_; }
    
    // Field algebra
    Field& operator+=(const Field& rhs);
    Field& operator-=(const Field& rhs);
    Field& operator*=(Real scalar);
    Field& operator/=(Real scalar);
    
    Field operator+(const Field& rhs) const;
    Field operator-(const Field& rhs) const;
    Field operator*(Real scalar) const;
    Field operator/(Real scalar) const;
    
    // Norms and statistics
    Real L1Norm() const;
    Real L2Norm() const;
    Real LinfNorm() const;
    T min() const;
    T max() const;
    T average() const;
    
    // Boundary update
    void updateBoundaryConditions();
    void correctBoundaryConditions();
    
    // Parallel communication
    void updateGhostCells();
    void syncProcessorPatches();
    
private:
    std::string name_;
    FieldType type_;
    const Mesh* mesh_;
    
    // Cell-centered data
    AlignedVector<T> data_;
    
    // Face-centered data (for flux computation)
    AlignedVector<T> faceData_;
    
    // Time levels
    SharedPtr<Field> oldTime_;
    SharedPtr<Field> oldOldTime_;
    
    // Boundary conditions
    boundary_map boundaryConditions_;
    
    // Helper template for gradient type
    template<typename U>
    struct GradientType {
        using type = void;
    };
    
    template<>
    struct GradientType<Real> {
        using type = Vector3;
    };
    
    template<>
    struct GradientType<Vector3> {
        using type = Matrix3;
    };
};

// Type aliases for common field types
using ScalarField = Field<Real>;
using VectorField = Field<Vector3>;
using TensorField = Field<Matrix3>;

// Field factory functions
template<typename T>
SharedPtr<Field<T>> createField(const Mesh& mesh, const std::string& name, 
                               const T& initialValue = T()) {
    return std::make_shared<Field<T>>(mesh, name, initialValue);
}

// Global field operations
template<typename T>
Real innerProduct(const Field<T>& a, const Field<T>& b) {
    Real sum = 0.0;
    for (Index i = 0; i < a.mesh().numCells(); ++i) {
        sum += dot(a[i], b[i]) * a.mesh().cell(i).volume();
    }
    return sum;
}

template<typename T>
Field<T> operator*(Real scalar, const Field<T>& field) {
    return field * scalar;
}

} // namespace cfd