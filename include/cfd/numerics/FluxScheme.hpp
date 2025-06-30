#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/numerics/Limiter.hpp"
#include <array>

namespace cfd::numerics {

// State vector for compressible flow
struct CompressibleState {
    Real rho;      // Density
    Vector3 rhoU;  // Momentum
    Real rhoE;     // Total energy
    
    // Primitive variables
    Vector3 velocity() const { return rhoU / rho; }
    Real pressure(Real gamma) const;
    Real temperature(Real gamma, Real R) const;
    Real soundSpeed(Real gamma) const;
    
    // Conservative to primitive conversion
    void toPrimitive(Real& p, Vector3& u, Real& T, Real gamma, Real R) const;
    static CompressibleState fromPrimitive(Real rho, const Vector3& u, Real p, Real gamma);
};

// Base flux scheme interface
template<typename StateType>
class FluxScheme {
public:
    FluxScheme(const Mesh& mesh) : mesh_(mesh) {}
    virtual ~FluxScheme() = default;
    
    // Compute fluxes for all faces
    virtual void computeFluxes(const Field<StateType>& state,
                              Field<StateType>& fluxes) = 0;
    
    // Set limiter for high-order schemes
    void setLimiter(SharedPtr<Limiter<StateType>> limiter) {
        limiter_ = limiter;
    }
    
protected:
    const Mesh& mesh_;
    SharedPtr<Limiter<StateType>> limiter_;
    
    // Compute flux at a single face
    virtual StateType computeFaceFlux(const StateType& stateL,
                                     const StateType& stateR,
                                     const Vector3& normal) const = 0;
};

// Upwind flux scheme
template<typename StateType>
class UpwindFlux : public FluxScheme<StateType> {
public:
    using FluxScheme<StateType>::FluxScheme;
    
    void computeFluxes(const Field<StateType>& state,
                      Field<StateType>& fluxes) override;
    
protected:
    StateType computeFaceFlux(const StateType& stateL,
                             const StateType& stateR,
                             const Vector3& normal) const override;
};

// Central difference flux
template<typename StateType>
class CentralFlux : public FluxScheme<StateType> {
public:
    using FluxScheme<StateType>::FluxScheme;
    
    void computeFluxes(const Field<StateType>& state,
                      Field<StateType>& fluxes) override;
    
protected:
    StateType computeFaceFlux(const StateType& stateL,
                             const StateType& stateR,
                             const Vector3& normal) const override;
};

// Roe flux for compressible flow
class RoeFlux : public FluxScheme<CompressibleState> {
public:
    RoeFlux(const Mesh& mesh, Real gamma) 
        : FluxScheme<CompressibleState>(mesh), gamma_(gamma) {}
    
    void computeFluxes(const Field<CompressibleState>& state,
                      Field<CompressibleState>& fluxes) override;
    
protected:
    CompressibleState computeFaceFlux(const CompressibleState& stateL,
                                     const CompressibleState& stateR,
                                     const Vector3& normal) const override;
    
private:
    Real gamma_;
    
    // Roe average state
    struct RoeAverage {
        Real rho;
        Vector3 u;
        Real H;
        Real c;
    };
    
    RoeAverage computeRoeAverage(const CompressibleState& stateL,
                                const CompressibleState& stateR) const;
    
    // Eigenvalues and eigenvectors
    std::array<Real, 5> eigenvalues(const RoeAverage& avg,
                                   const Vector3& normal) const;
    
    void eigendecomposition(const RoeAverage& avg,
                           const Vector3& normal,
                           Matrix<Real, 5, 5>& R,
                           Matrix<Real, 5, 5>& L) const;
};

// HLLC (Harten-Lax-van Leer-Contact) flux
class HLLCFlux : public FluxScheme<CompressibleState> {
public:
    HLLCFlux(const Mesh& mesh, Real gamma)
        : FluxScheme<CompressibleState>(mesh), gamma_(gamma) {}
    
    void computeFluxes(const Field<CompressibleState>& state,
                      Field<CompressibleState>& fluxes) override;
    
protected:
    CompressibleState computeFaceFlux(const CompressibleState& stateL,
                                     const CompressibleState& stateR,
                                     const Vector3& normal) const override;
    
private:
    Real gamma_;
    
    // Wave speed estimates
    struct WaveSpeeds {
        Real SL, SR, SM;  // Left, right, and middle wave speeds
    };
    
    WaveSpeeds estimateWaveSpeeds(const CompressibleState& stateL,
                                  const CompressibleState& stateR,
                                  const Vector3& normal) const;
    
    // HLLC flux computation
    CompressibleState computeHLLCFlux(const CompressibleState& stateL,
                                     const CompressibleState& stateR,
                                     const WaveSpeeds& speeds,
                                     const Vector3& normal) const;
};

// AUSM (Advection Upstream Splitting Method) flux
class AUSMFlux : public FluxScheme<CompressibleState> {
public:
    AUSMFlux(const Mesh& mesh, Real gamma)
        : FluxScheme<CompressibleState>(mesh), gamma_(gamma) {}
    
    void computeFluxes(const Field<CompressibleState>& state,
                      Field<CompressibleState>& fluxes) override;
    
protected:
    CompressibleState computeFaceFlux(const CompressibleState& stateL,
                                     const CompressibleState& stateR,
                                     const Vector3& normal) const override;
    
private:
    Real gamma_;
    
    // AUSM specific functions
    Real machSplitPlus(Real M) const;
    Real machSplitMinus(Real M) const;
    Real pressureSplitPlus(Real M) const;
    Real pressureSplitMinus(Real M) const;
};

// MUSCL (Monotonic Upstream-Centered Scheme for Conservation Laws) reconstruction
template<typename StateType>
class MUSCLReconstruction {
public:
    MUSCLReconstruction(const Mesh& mesh, SharedPtr<Limiter<StateType>> limiter)
        : mesh_(mesh), limiter_(limiter) {}
    
    // Reconstruct states at face
    void reconstruct(const Field<StateType>& state,
                    Index faceId,
                    StateType& stateL,
                    StateType& stateR) const;
    
private:
    const Mesh& mesh_;
    SharedPtr<Limiter<StateType>> limiter_;
    
    // Compute limited slopes
    StateType computeLimitedSlope(const StateType& stateC,
                                 const StateType& stateL,
                                 const StateType& stateR) const;
};

// WENO5 (5th-order Weighted Essentially Non-Oscillatory) reconstruction
template<typename StateType>
class WENO5Reconstruction {
public:
    WENO5Reconstruction(const Mesh& mesh) : mesh_(mesh) {}
    
    // High-order reconstruction
    void reconstruct(const Field<StateType>& state,
                    Index faceId,
                    StateType& stateL,
                    StateType& stateR) const;
    
private:
    const Mesh& mesh_;
    static constexpr Real epsilon_ = 1e-6;
    
    // WENO weights computation
    std::array<Real, 3> computeWeights(const std::array<Real, 5>& stencil) const;
    Real smoothnessIndicator(Real vm, Real v0, Real vp) const;
};

// Factory function
template<typename StateType>
SharedPtr<FluxScheme<StateType>> createFluxScheme(FluxScheme type,
                                                  const Mesh& mesh,
                                                  Real gamma = 1.4) {
    switch (type) {
        case FluxScheme::UPWIND:
            return std::make_shared<UpwindFlux<StateType>>(mesh);
        case FluxScheme::CENTRAL:
            return std::make_shared<CentralFlux<StateType>>(mesh);
        case FluxScheme::ROE:
            if constexpr (std::is_same_v<StateType, CompressibleState>) {
                return std::make_shared<RoeFlux>(mesh, gamma);
            }
            throw std::runtime_error("Roe flux only for compressible flow");
        case FluxScheme::HLLC:
            if constexpr (std::is_same_v<StateType, CompressibleState>) {
                return std::make_shared<HLLCFlux>(mesh, gamma);
            }
            throw std::runtime_error("HLLC flux only for compressible flow");
        case FluxScheme::AUSM:
            if constexpr (std::is_same_v<StateType, CompressibleState>) {
                return std::make_shared<AUSMFlux>(mesh, gamma);
            }
            throw std::runtime_error("AUSM flux only for compressible flow");
        default:
            throw std::runtime_error("Unknown flux scheme");
    }
}

} // namespace cfd::numerics