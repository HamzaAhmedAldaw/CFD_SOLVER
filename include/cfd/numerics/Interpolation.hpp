#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/core/Mesh.hpp"
#include "cfd/core/Face.hpp"

namespace cfd::numerics {

// Base interpolation class
template<typename T>
class InterpolationScheme {
public:
    InterpolationScheme(const Mesh& mesh) : mesh_(mesh) {}
    virtual ~InterpolationScheme() = default;
    
    // Interpolate from cells to face
    virtual T interpolate(const Field<T>& phi, const Face& face) const = 0;
    
    // Interpolate all faces
    virtual void interpolateToFaces(const Field<T>& phi,
                                   Field<T>& phiFace) const {
        for (Index faceId = 0; faceId < mesh_.numFaces(); ++faceId) {
            const Face& face = mesh_.face(faceId);
            phiFace.face(faceId) = interpolate(phi, face);
        }
    }
    
protected:
    const Mesh& mesh_;
};

// Linear interpolation (central differencing)
template<typename T>
class LinearInterpolation : public InterpolationScheme<T> {
public:
    using InterpolationScheme<T>::InterpolationScheme;
    
    T interpolate(const Field<T>& phi, const Face& face) const override {
        if (face.isInternal()) {
            // Linear interpolation between owner and neighbour
            return face.weight() * phi[face.neighbour()] + 
                   (1.0 - face.weight()) * phi[face.owner()];
        } else {
            // Boundary face - use boundary condition
            return phi.face(face.id());
        }
    }
};

// Upwind interpolation
template<typename T>
class UpwindInterpolation : public InterpolationScheme<T> {
public:
    UpwindInterpolation(const Mesh& mesh, const ScalarField& flux)
        : InterpolationScheme<T>(mesh), flux_(flux) {}
    
    T interpolate(const Field<T>& phi, const Face& face) const override {
        if (face.isInternal()) {
            // Select upwind value based on flux direction
            Real phiFace = flux_.face(face.id());
            if (phiFace >= 0) {
                return phi[face.owner()];
            } else {
                return phi[face.neighbour()];
            }
        } else {
            return phi.face(face.id());
        }
    }
    
private:
    const ScalarField& flux_;
};

// TVD (Total Variation Diminishing) interpolation
template<typename T>
class TVDInterpolation : public InterpolationScheme<T> {
public:
    enum LimiterType {
        MINMOD,
        VAN_LEER,
        SUPERBEE,
        MUSCL,
        QUICK
    };
    
    TVDInterpolation(const Mesh& mesh, const ScalarField& flux,
                     LimiterType limiter = VAN_LEER)
        : InterpolationScheme<T>(mesh), flux_(flux), limiterType_(limiter) {}
    
    T interpolate(const Field<T>& phi, const Face& face) const override;
    
private:
    const ScalarField& flux_;
    LimiterType limiterType_;
    
    // Limiter functions
    Real limiterFunction(Real r) const;
    Real minmod(Real a, Real b) const { return sign(a) * std::max(Real(0), std::min(std::abs(a), sign(a)*b)); }
    Real vanLeer(Real r) const { return (r + std::abs(r)) / (1 + std::abs(r)); }
    Real superbee(Real r) const { return std::max(std::min(2*r, Real(1)), std::min(r, Real(2))); }
    
    // Compute gradient ratio
    Real computeGradientRatio(const Field<T>& phi, const Face& face) const;
};

// High-Resolution Scheme (HRS) interpolation
template<typename T>
class HRSInterpolation : public InterpolationScheme<T> {
public:
    HRSInterpolation(const Mesh& mesh, const ScalarField& flux,
                     Real blendingFactor = 1.0)
        : InterpolationScheme<T>(mesh), flux_(flux), 
          beta_(blendingFactor), tvd_(mesh, flux) {}
    
    T interpolate(const Field<T>& phi, const Face& face) const override {
        // Blend between upwind and high-order scheme
        T upwindValue = upwindInterpolate(phi, face);
        T highOrderValue = tvd_.interpolate(phi, face);
        
        return (1.0 - beta_) * upwindValue + beta_ * highOrderValue;
    }
    
private:
    const ScalarField& flux_;
    Real beta_;  // Blending factor
    TVDInterpolation<T> tvd_;
    
    T upwindInterpolate(const Field<T>& phi, const Face& face) const;
};

// Harmonic interpolation (for diffusion coefficients)
class HarmonicInterpolation : public InterpolationScheme<Real> {
public:
    using InterpolationScheme<Real>::InterpolationScheme;
    
    Real interpolate(const Field<Real>& phi, const Face& face) const override {
        if (face.isInternal()) {
            Real phiOwner = phi[face.owner()];
            Real phiNeighbour = phi[face.neighbour()];
            
            // Harmonic mean
            return 2.0 * phiOwner * phiNeighbour / 
                   (phiOwner + phiNeighbour + SMALL);
        } else {
            return phi.face(face.id());
        }
    }
};

// Limited linear interpolation
template<typename T>
class LimitedLinearInterpolation : public InterpolationScheme<T> {
public:
    LimitedLinearInterpolation(const Mesh& mesh, Real psi = 1.0)
        : InterpolationScheme<T>(mesh), psi_(psi) {}
    
    T interpolate(const Field<T>& phi, const Face& face) const override;
    
private:
    Real psi_;  // Limiter value [0,1]
    
    // Compute limited value
    T limitedValue(const T& phiCD, const T& phiUD, const Face& face) const;
};

// NVD (Normalised Variable Diagram) schemes
template<typename T>
class NVDInterpolation : public InterpolationScheme<T> {
public:
    enum SchemeType {
        GAMMA,     // Jasak's Gamma scheme
        SFCD,      // Self-Filtered Central Differencing
        CUBISTA    // Convergent and Universally Bounded Interpolation Scheme
    };
    
    NVDInterpolation(const Mesh& mesh, const ScalarField& flux,
                     SchemeType scheme = GAMMA, Real beta = 0.1)
        : InterpolationScheme<T>(mesh), flux_(flux), 
          scheme_(scheme), beta_(beta) {}
    
    T interpolate(const Field<T>& phi, const Face& face) const override;
    
private:
    const ScalarField& flux_;
    SchemeType scheme_;
    Real beta_;  // Scheme parameter
    
    // NVD functions
    Real gammaFunction(Real phiTilde) const;
    Real sfcdFunction(Real phiTilde) const;
    Real cubistaFunction(Real phiTilde) const;
    
    // Normalized variable
    Real normalizedVariable(const Field<T>& phi, const Face& face) const;
};

// Quadratic upstream interpolation (QUICK)
template<typename T>
class QUICKInterpolation : public InterpolationScheme<T> {
public:
    QUICKInterpolation(const Mesh& mesh, const ScalarField& flux)
        : InterpolationScheme<T>(mesh), flux_(flux) {}
    
    T interpolate(const Field<T>& phi, const Face& face) const override;
    
private:
    const ScalarField& flux_;
    
    // Find upstream cells for quadratic interpolation
    std::array<Index, 3> findUpstreamCells(const Face& face) const;
};

// Factory function
template<typename T>
SharedPtr<InterpolationScheme<T>> createInterpolationScheme(
    const std::string& type,
    const Mesh& mesh,
    const ScalarField* flux = nullptr) {
    
    if (type == "linear") {
        return std::make_shared<LinearInterpolation<T>>(mesh);
    } else if (type == "upwind" && flux) {
        return std::make_shared<UpwindInterpolation<T>>(mesh, *flux);
    } else if (type == "limitedLinear") {
        return std::make_shared<LimitedLinearInterpolation<T>>(mesh);
    } else if (type == "vanLeer" && flux) {
        return std::make_shared<TVDInterpolation<T>>(mesh, *flux, 
            TVDInterpolation<T>::VAN_LEER);
    } else if (type == "QUICK" && flux) {
        return std::make_shared<QUICKInterpolation<T>>(mesh, *flux);
    } else {
        throw std::runtime_error("Unknown interpolation scheme: " + type);
    }
}

} // namespace cfd::numerics