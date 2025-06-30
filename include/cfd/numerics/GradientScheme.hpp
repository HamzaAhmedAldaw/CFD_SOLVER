#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/core/Mesh.hpp"

namespace cfd::numerics {

// Base gradient scheme interface
class GradientScheme {
public:
    GradientScheme(const Mesh& mesh) : mesh_(mesh) {}
    virtual ~GradientScheme() = default;
    
    // Compute gradient of scalar field
    virtual void compute(const ScalarField& phi,
                        VectorField& gradPhi) const = 0;
    
    // Compute gradient of vector field
    virtual void compute(const VectorField& U,
                        TensorField& gradU) const = 0;
    
protected:
    const Mesh& mesh_;
};

// Green-Gauss gradient computation
class GreenGaussGradient : public GradientScheme {
public:
    enum Variant {
        CELL_BASED,    // Use cell values for face interpolation
        NODE_BASED     // Use node values (more accurate, more expensive)
    };
    
    GreenGaussGradient(const Mesh& mesh, Variant variant = CELL_BASED)
        : GradientScheme(mesh), variant_(variant) {}
    
    void compute(const ScalarField& phi, VectorField& gradPhi) const override;
    void compute(const VectorField& U, TensorField& gradU) const override;
    
private:
    Variant variant_;
    
    // Helper functions
    void computeCellBased(const ScalarField& phi, VectorField& gradPhi) const;
    void computeNodeBased(const ScalarField& phi, VectorField& gradPhi) const;
    
    // Node value computation
    Real interpolateToNode(const ScalarField& phi, Index nodeId) const;
    Vector3 interpolateToNode(const VectorField& U, Index nodeId) const;
};

// Least-squares gradient computation
class LeastSquaresGradient : public GradientScheme {
public:
    enum WeightingType {
        UNIFORM,           // No weighting
        INVERSE_DISTANCE,  // 1/|d| weighting
        INVERSE_DISTANCE_SQUARED  // 1/|d|² weighting
    };
    
    LeastSquaresGradient(const Mesh& mesh, 
                        WeightingType weighting = INVERSE_DISTANCE)
        : GradientScheme(mesh), weighting_(weighting) {
        precomputeMatrices();
    }
    
    void compute(const ScalarField& phi, VectorField& gradPhi) const override;
    void compute(const VectorField& U, TensorField& gradU) const override;
    
private:
    WeightingType weighting_;
    
    // Pre-computed least-squares matrices for efficiency
    std::vector<Matrix3> lsqMatrices_;
    std::vector<std::vector<Real>> weights_;
    
    void precomputeMatrices();
    Real computeWeight(Real distance) const;
};

// Weighted least-squares with iterative limiting
class WeightedLeastSquaresGradient : public LeastSquaresGradient {
public:
    WeightedLeastSquaresGradient(const Mesh& mesh, int iterations = 2)
        : LeastSquaresGradient(mesh, INVERSE_DISTANCE_SQUARED),
          iterations_(iterations) {}
    
    void compute(const ScalarField& phi, VectorField& gradPhi) const override;
    void compute(const VectorField& U, TensorField& gradU) const override;
    
private:
    int iterations_;
    
    // Iterative weight adjustment based on gradient magnitude
    void adjustWeights(const VectorField& gradPhi,
                      std::vector<std::vector<Real>>& weights) const;
};

// Gradient limiter for stability
template<typename T>
class GradientLimiter {
public:
    GradientLimiter(const Mesh& mesh) : mesh_(mesh) {}
    virtual ~GradientLimiter() = default;
    
    // Limit gradient field
    virtual void limit(Field<T>& phi, 
                      Field<typename GradientType<T>::type>& gradPhi) const = 0;
    
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
};

// Barth-Jespersen limiter
template<typename T>
class BarthJespersenLimiter : public GradientLimiter<T> {
public:
    using GradientLimiter<T>::GradientLimiter;
    
    void limit(Field<T>& phi,
              Field<typename GradientLimiter<T>::template GradientType<T>::type>& gradPhi) const override;
    
private:
    T computeLimiter(Index cellId, const T& phiMin, const T& phiMax,
                    const T& phiCell, const Vector3& grad) const;
};

// Venkatakrishnan limiter
template<typename T>
class VenkatakrishnanLimiter : public GradientLimiter<T> {
public:
    VenkatakrishnanLimiter(const Mesh& mesh, Real K = 5.0)
        : GradientLimiter<T>(mesh), K_(K) {
        computeEpsilon();
    }
    
    void limit(Field<T>& phi,
              Field<typename GradientLimiter<T>::template GradientType<T>::type>& gradPhi) const override;
    
private:
    Real K_;
    std::vector<Real> epsilon_;  // Cell-based epsilon values
    
    void computeEpsilon();
    Real smoothFunction(Real r) const;
};

// Multi-directional gradient limiter
template<typename T>
class MultiDirectionalLimiter : public GradientLimiter<T> {
public:
    using GradientLimiter<T>::GradientLimiter;
    
    void limit(Field<T>& phi,
              Field<typename GradientLimiter<T>::template GradientType<T>::type>& gradPhi) const override;
    
private:
    // Limit in each face direction separately
    std::vector<Real> computeFaceLimiters(Index cellId, const Field<T>& phi,
                                          const Vector3& grad) const;
};

// Factory function
inline SharedPtr<GradientScheme> createGradientScheme(GradientScheme type,
                                                     const Mesh& mesh) {
    switch (type) {
        case GradientScheme::GREEN_GAUSS:
            return std::make_shared<GreenGaussGradient>(mesh);
        case GradientScheme::LEAST_SQUARES:
            return std::make_shared<LeastSquaresGradient>(mesh);
        case GradientScheme::WEIGHTED_LEAST_SQUARES:
            return std::make_shared<WeightedLeastSquaresGradient>(mesh);
        default:
            throw std::runtime_error("Unknown gradient scheme");
    }
}

} // namespace cfd::numerics