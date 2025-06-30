#pragma once

#include "cfd/core/Types.hpp"
#include <functional>
#include <memory>

namespace cfd {

template<typename T>
class BoundaryCondition {
public:
    BoundaryCondition(BCType type) : type_(type) {}
    virtual ~BoundaryCondition() = default;
    
    // Type of boundary condition
    BCType type() const { return type_; }
    
    // Get boundary value
    virtual T value(const T& cellValue, const Vector3& normal,
                   const Vector3& position, Real time) const = 0;
    
    // Get gradient at boundary
    virtual T gradient(const T& cellValue, const Vector3& normal,
                      const Vector3& position, Real time) const = 0;
    
    // Update coefficients for implicit treatment
    virtual void updateCoeffs(Real& diag, T& source,
                             const Vector3& normal, Real area) const = 0;
    
    // Clone for deep copy
    virtual SharedPtr<BoundaryCondition<T>> clone() const = 0;
    
protected:
    BCType type_;
};

// Fixed value (Dirichlet) boundary condition
template<typename T>
class FixedValueBC : public BoundaryCondition<T> {
public:
    FixedValueBC(const T& value)
        : BoundaryCondition<T>(BCType::DIRICHLET), value_(value) {}
    
    FixedValueBC(std::function<T(const Vector3&, Real)> valueFunc)
        : BoundaryCondition<T>(BCType::DIRICHLET), valueFunc_(valueFunc) {}
    
    T value(const T& /*cellValue*/, const Vector3& /*normal*/,
           const Vector3& position, Real time) const override {
        if (valueFunc_) {
            return valueFunc_(position, time);
        }
        return value_;
    }
    
    T gradient(const T& cellValue, const Vector3& normal,
              const Vector3& position, Real time) const override {
        // For Dirichlet BC, gradient computed from cell value and face value
        T faceValue = value(cellValue, normal, position, time);
        return (faceValue - cellValue) * 2.0; // Simplified - needs proper distance
    }
    
    void updateCoeffs(Real& diag, T& source,
                     const Vector3& normal, Real area) const override {
        // For implicit discretization
        diag += area;
        source += value_ * area;
    }
    
    SharedPtr<BoundaryCondition<T>> clone() const override {
        return std::make_shared<FixedValueBC<T>>(*this);
    }
    
private:
    T value_;
    std::function<T(const Vector3&, Real)> valueFunc_;
};

// Fixed gradient (Neumann) boundary condition
template<typename T>
class FixedGradientBC : public BoundaryCondition<T> {
public:
    FixedGradientBC(const T& gradient = T())
        : BoundaryCondition<T>(BCType::NEUMANN), gradient_(gradient) {}
    
    FixedGradientBC(std::function<T(const Vector3&, Real)> gradFunc)
        : BoundaryCondition<T>(BCType::NEUMANN), gradFunc_(gradFunc) {}
    
    T value(const T& cellValue, const Vector3& normal,
           const Vector3& position, Real time) const override {
        T grad = gradient(cellValue, normal, position, time);
        return cellValue + grad * 0.5; // Simplified - needs proper distance
    }
    
    T gradient(const T& /*cellValue*/, const Vector3& /*normal*/,
              const Vector3& position, Real time) const override {
        if (gradFunc_) {
            return gradFunc_(position, time);
        }
        return gradient_;
    }
    
    void updateCoeffs(Real& /*diag*/, T& source,
                     const Vector3& normal, Real area) const override {
        source += gradient_ * area;
    }
    
    SharedPtr<BoundaryCondition<T>> clone() const override {
        return std::make_shared<FixedGradientBC<T>>(*this);
    }
    
private:
    T gradient_;
    std::function<T(const Vector3&, Real)> gradFunc_;
};

// Zero gradient boundary condition
template<typename T>
class ZeroGradientBC : public FixedGradientBC<T> {
public:
    ZeroGradientBC() : FixedGradientBC<T>(T()) {}
    
    SharedPtr<BoundaryCondition<T>> clone() const override {
        return std::make_shared<ZeroGradientBC<T>>(*this);
    }
};

// Mixed (Robin) boundary condition: a*phi + b*dphi/dn = c
template<typename T>
class MixedBC : public BoundaryCondition<T> {
public:
    MixedBC(Real a, Real b, const T& c)
        : BoundaryCondition<T>(BCType::ROBIN), a_(a), b_(b), c_(c) {}
    
    T value(const T& cellValue, const Vector3& normal,
           const Vector3& position, Real time) const override {
        // Solve a*phi + b*dphi/dn = c for phi
        Real delta = 0.5; // Simplified distance
        return (c_ + b_ * cellValue / delta) / (a_ + b_ / delta);
    }
    
    T gradient(const T& cellValue, const Vector3& normal,
              const Vector3& position, Real time) const override {
        T faceValue = value(cellValue, normal, position, time);
        return (c_ - a_ * faceValue) / b_;
    }
    
    void updateCoeffs(Real& diag, T& source,
                     const Vector3& normal, Real area) const override {
        Real delta = 0.5; // Simplified
        diag += a_ * area / (a_ + b_ / delta);
        source += c_ * area / (a_ + b_ / delta);
    }
    
    SharedPtr<BoundaryCondition<T>> clone() const override {
        return std::make_shared<MixedBC<T>>(*this);
    }
    
private:
    Real a_, b_;
    T c_;
};

// Inlet boundary condition
template<typename T>
class InletBC : public FixedValueBC<T> {
public:
    InletBC(const T& value) 
        : FixedValueBC<T>(value) {
        this->type_ = BCType::INLET;
    }
    
    SharedPtr<BoundaryCondition<T>> clone() const override {
        return std::make_shared<InletBC<T>>(*this);
    }
};

// Outlet boundary condition (usually zero gradient)
template<typename T>
class OutletBC : public ZeroGradientBC<T> {
public:
    OutletBC() {
        this->type_ = BCType::OUTLET;
    }
    
    SharedPtr<BoundaryCondition<T>> clone() const override {
        return std::make_shared<OutletBC<T>>(*this);
    }
};

// Wall boundary conditions
template<typename T>
class WallBC : public BoundaryCondition<T> {
public:
    WallBC() : BoundaryCondition<T>(BCType::WALL) {}
    
    SharedPtr<BoundaryCondition<T>> clone() const override {
        return std::make_shared<WallBC<T>>(*this);
    }
};

// No-slip wall for velocity
template<>
class WallBC<Vector3> : public FixedValueBC<Vector3> {
public:
    WallBC(const Vector3& wallVelocity = Vector3::Zero())
        : FixedValueBC<Vector3>(wallVelocity) {
        this->type_ = BCType::WALL;
    }
    
    SharedPtr<BoundaryCondition<Vector3>> clone() const override {
        return std::make_shared<WallBC<Vector3>>(*this);
    }
};

// Slip wall for velocity
class SlipWallBC : public BoundaryCondition<Vector3> {
public:
    SlipWallBC() : BoundaryCondition<Vector3>(BCType::SYMMETRY) {}
    
    Vector3 value(const Vector3& cellValue, const Vector3& normal,
                  const Vector3& /*position*/, Real /*time*/) const override {
        // Remove normal component
        return cellValue - normal * cellValue.dot(normal);
    }
    
    Vector3 gradient(const Vector3& /*cellValue*/, const Vector3& /*normal*/,
                    const Vector3& /*position*/, Real /*time*/) const override {
        return Vector3::Zero();
    }
    
    void updateCoeffs(Real& /*diag*/, Vector3& /*source*/,
                     const Vector3& /*normal*/, Real /*area*/) const override {
        // Handled explicitly
    }
    
    SharedPtr<BoundaryCondition<Vector3>> clone() const override {
        return std::make_shared<SlipWallBC>(*this);
    }
};

// Factory functions
template<typename T>
SharedPtr<BoundaryCondition<T>> createBC(BCType type, const T& value = T()) {
    switch (type) {
        case BCType::DIRICHLET:
            return std::make_shared<FixedValueBC<T>>(value);
        case BCType::NEUMANN:
            return std::make_shared<FixedGradientBC<T>>(value);
        case BCType::INLET:
            return std::make_shared<InletBC<T>>(value);
        case BCType::OUTLET:
            return std::make_shared<OutletBC<T>>();
        default:
            throw std::runtime_error("Unsupported BC type");
    }
}

} // namespace cfd