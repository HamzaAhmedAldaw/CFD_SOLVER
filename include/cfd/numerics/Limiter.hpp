#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/core/Mesh.hpp"
#include <algorithm>
#include <cmath>

namespace cfd::numerics {

// Base limiter class
template<typename T>
class Limiter {
public:
    Limiter(const Mesh& mesh) : mesh_(mesh) {}
    virtual ~Limiter() = default;
    
    // Apply limiter to field gradients
    virtual void limit(const Field<T>& phi,
                      Field<typename GradientType<T>::type>& gradPhi) const = 0;
    
    // Compute limiter value for a single cell
    virtual Real computeLimiter(Index cellId,
                               const Field<T>& phi,
                               const typename GradientType<T>::type& grad) const = 0;
    
protected:
    const Mesh& mesh_;
    
    // Helper for gradient types
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
    
    // Find min/max values in neighborhood
    std::pair<T, T> findMinMax(Index cellId, const Field<T>& phi) const;
};

// Barth-Jespersen limiter
template<typename T>
class BarthJespersenLimiter : public Limiter<T> {
public:
    using typename Limiter<T>::GradientType;
    using Limiter<T>::Limiter;
    
    void limit(const Field<T>& phi,
              Field<typename GradientType<T>::type>& gradPhi) const override;
    
    Real computeLimiter(Index cellId,
                       const Field<T>& phi,
                       const typename GradientType<T>::type& grad) const override;
};

// Venkatakrishnan limiter
template<typename T>
class VenkatakrishnanLimiter : public Limiter<T> {
public:
    using typename Limiter<T>::GradientType;
    
    VenkatakrishnanLimiter(const Mesh& mesh, Real K = 5.0)
        : Limiter<T>(mesh), K_(K) {
        computeEpsilon();
    }
    
    void limit(const Field<T>& phi,
              Field<typename GradientType<T>::type>& gradPhi) const override;
    
    Real computeLimiter(Index cellId,
                       const Field<T>& phi,
                       const typename GradientType<T>::type& grad) const override;
    
private:
    Real K_;  // Venkatakrishnan constant
    std::vector<Real> epsilon_;  // Cell-based epsilon values
    
    void computeEpsilon();
    Real smoothFunction(Real y2, Real eps2) const;
};

// Minmod limiter
template<typename T>
class MinmodLimiter : public Limiter<T> {
public:
    using typename Limiter<T>::GradientType;
    using Limiter<T>::Limiter;
    
    void limit(const Field<T>& phi,
              Field<typename GradientType<T>::type>& gradPhi) const override;
    
    Real computeLimiter(Index cellId,
                       const Field<T>& phi,
                       const typename GradientType<T>::type& grad) const override;
    
private:
    Real minmod(Real a, Real b) const {
        return sign(a) * std::max(Real(0), std::min(std::abs(a), sign(a) * b));
    }
    
    Real minmod(Real a, Real b, Real c) const {
        return sign(a) * std::max(Real(0), 
            std::min({std::abs(a), sign(a) * b, sign(a) * c}));
    }
};

// Van Leer limiter
template<typename T>
class VanLeerLimiter : public Limiter<T> {
public:
    using typename Limiter<T>::GradientType;
    using Limiter<T>::Limiter;
    
    void limit(const Field<T>& phi,
              Field<typename GradientType<T>::type>& gradPhi) const override;
    
    Real computeLimiter(Index cellId,
                       const Field<T>& phi,
                       const typename GradientType<T>::type& grad) const override;
    
private:
    Real vanLeerFunction(Real r) const {
        return (r + std::abs(r)) / (1.0 + std::abs(r));
    }
};

// Superbee limiter
template<typename T>
class SuperbeeLimiter : public Limiter<T> {
public:
    using typename Limiter<T>::GradientType;
    using Limiter<T>::Limiter;
    
    void limit(const Field<T>& phi,
              Field<typename GradientType<T>::type>& gradPhi) const override;
    
    Real computeLimiter(Index cellId,
                       const Field<T>& phi,
                       const typename GradientType<T>::type& grad) const override;
    
private:
    Real superbeeFunction(Real r) const {
        return std::max({Real(0), std::min(2*r, Real(1)), std::min(r, Real(2))});
    }
};

// Multi-dimensional limiter (MDL)
template<typename T>
class MultiDimensionalLimiter : public Limiter<T> {
public:
    using typename Limiter<T>::GradientType;
    
    MultiDimensionalLimiter(const Mesh& mesh, 
                           SharedPtr<Limiter<T>> baseLimiter)
        : Limiter<T>(mesh), baseLimiter_(baseLimiter) {}
    
    void limit(const Field<T>& phi,
              Field<typename GradientType<T>::type>& gradPhi) const override;
    
    Real computeLimiter(Index cellId,
                       const Field<T>& phi,
                       const typename GradientType<T>::type& grad) const override;
    
private:
    SharedPtr<Limiter<T>> baseLimiter_;
    
    // Compute directional limiters
    std::vector<Real> computeDirectionalLimiters(
        Index cellId,
        const Field<T>& phi,
        const typename GradientType<T>::type& grad) const;
};

// WENO limiter for high-order schemes
template<typename T>
class WENOLimiter : public Limiter<T> {
public:
    using typename Limiter<T>::GradientType;
    
    WENOLimiter(const Mesh& mesh, int order = 5)
        : Limiter<T>(mesh), order_(order) {}
    
    void limit(const Field<T>& phi,
              Field<typename GradientType<T>::type>& gradPhi) const override;
    
    Real computeLimiter(Index cellId,
                       const Field<T>& phi,
                       const typename GradientType<T>::type& grad) const override;
    
private:
    int order_;
    static constexpr Real epsilon_ = 1e-6;
    
    // WENO reconstruction
    T reconstructWENO(const std::vector<T>& stencil) const;
    
    // Smoothness indicators
    Real smoothnessIndicator(const std::vector<Real>& values) const;
    
    // WENO weights
    std::vector<Real> computeWeights(const std::vector<Real>& beta) const;
};

// Slope limiter for systems (e.g., compressible flow)
template<typename StateType>
class SystemLimiter {
public:
    SystemLimiter(const Mesh& mesh,
                  SharedPtr<Limiter<Real>> scalarLimiter)
        : mesh_(mesh), scalarLimiter_(scalarLimiter) {}
    
    // Apply limiter to each component
    void limit(const Field<StateType>& state,
              Field<typename StateGradient<StateType>::type>& gradState) const;
    
private:
    const Mesh& mesh_;
    SharedPtr<Limiter<Real>> scalarLimiter_;
    
    // Helper for state gradient types
    template<typename T>
    struct StateGradient {
        using type = void;
    };
};

// Implementation helper functions
template<typename T>
std::pair<T, T> Limiter<T>::findMinMax(Index cellId, const Field<T>& phi) const {
    const Cell& cell = mesh_.cell(cellId);
    
    T phiMin = phi[cellId];
    T phiMax = phi[cellId];
    
    // Check all neighbors
    for (Index nbrId : cell.neighbors()) {
        if constexpr (std::is_same_v<T, Real>) {
            phiMin = std::min(phiMin, phi[nbrId]);
            phiMax = std::max(phiMax, phi[nbrId]);
        } else if constexpr (std::is_same_v<T, Vector3>) {
            // Component-wise min/max for vectors
            for (int i = 0; i < 3; ++i) {
                phiMin[i] = std::min(phiMin[i], phi[nbrId][i]);
                phiMax[i] = std::max(phiMax[i], phi[nbrId][i]);
            }
        }
    }
    
    return {phiMin, phiMax};
}

// Factory function
template<typename T>
SharedPtr<Limiter<T>> createLimiter(LimiterType type,
                                   const Mesh& mesh,
                                   Real parameter = 1.0) {
    switch (type) {
        case LimiterType::NONE:
            return nullptr;
        case LimiterType::BARTH_JESPERSEN:
            return std::make_shared<BarthJespersenLimiter<T>>(mesh);
        case LimiterType::VENKATAKRISHNAN:
            return std::make_shared<VenkatakrishnanLimiter<T>>(mesh, parameter);
        case LimiterType::MINMOD:
            return std::make_shared<MinmodLimiter<T>>(mesh);
        case LimiterType::VANLEER:
            return std::make_shared<VanLeerLimiter<T>>(mesh);
        case LimiterType::SUPERBEE:
            return std::make_shared<SuperbeeLimiter<T>>(mesh);
        default:
            throw std::runtime_error("Unknown limiter type");
    }
}

} // namespace cfd::numerics