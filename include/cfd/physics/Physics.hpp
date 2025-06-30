#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/core/Mesh.hpp"
#include <memory>

namespace cfd::physics {

// Base physics model class
class PhysicsModel {
public:
    PhysicsModel(const Mesh& mesh) : mesh_(mesh) {}
    virtual ~PhysicsModel() = default;
    
    // Model name
    virtual std::string name() const = 0;
    
    // Initialize fields
    virtual void initialize() = 0;
    
    // Update model (e.g., properties, source terms)
    virtual void update() = 0;
    
    // Solve governing equations
    virtual void solve() = 0;
    
    // Time step constraint
    virtual Real maxTimeStep() const { return LARGE; }
    
protected:
    const Mesh& mesh_;
};

// Transport equation solver
template<typename T>
class TransportEquation {
public:
    TransportEquation(const Mesh& mesh, const std::string& name)
        : mesh_(mesh), equationName_(name) {}
    
    // Set equation components
    void setTransient(bool transient) { transient_ = transient; }
    void setConvection(bool convection) { convection_ = convection; }
    void setDiffusion(bool diffusion) { diffusion_ = diffusion; }
    
    // Set fields
    void setVelocity(SharedPtr<VectorField> U) { U_ = U; }
    void setDiffusivity(SharedPtr<ScalarField> gamma) { gamma_ = gamma; }
    void setSource(SharedPtr<Field<T>> source) { source_ = source; }
    
    // Solve transport equation: ∂φ/∂t + ∇·(Uφ) - ∇·(Γ∇φ) = S
    void solve(Field<T>& phi, Real dt);
    
    // Individual terms
    void addTemporalDerivative(Field<T>& phi, Real dt, SparseMatrix& A, VectorX& b);
    void addConvection(const Field<T>& phi, SparseMatrix& A, VectorX& b);
    void addDiffusion(const Field<T>& phi, SparseMatrix& A, VectorX& b);
    void addSource(const Field<T>& phi, VectorX& b);
    
    // Flux computation
    Real computeConvectiveFlux(const Face& face, const T& phiOwner, const T& phiNeighbour) const;
    Real computeDiffusiveFlux(const Face& face, const T& phiOwner, const T& phiNeighbour) const;
    
private:
    const Mesh& mesh_;
    std::string equationName_;
    
    // Equation flags
    bool transient_ = true;
    bool convection_ = true;
    bool diffusion_ = true;
    
    // Fields
    SharedPtr<VectorField> U_;
    SharedPtr<ScalarField> gamma_;
    SharedPtr<Field<T>> source_;
    
    // Discretization schemes
    SharedPtr<InterpolationScheme<T>> convectionScheme_;
    SharedPtr<GradientScheme> gradientScheme_;
};

// Conservation law in flux form: ∂U/∂t + ∇·F(U) = S
template<typename StateType>
class ConservationLaw : public PhysicsModel {
public:
    ConservationLaw(const Mesh& mesh) : PhysicsModel(mesh) {}
    
    // Flux function F(U)
    virtual StateType flux(const StateType& state, const Vector3& normal) const = 0;
    
    // Source term S(U)
    virtual StateType source(const StateType& state, const Vector3& position) const {
        return StateType();  // Default: no source
    }
    
    // Maximum wave speed (for CFL condition)
    virtual Real maxWaveSpeed(const StateType& state) const = 0;
    
    // Solve using finite volume method
    void solve() override;
    
protected:
    SharedPtr<Field<StateType>> state_;
    SharedPtr<FluxScheme<StateType>> fluxScheme_;
    SharedPtr<TimeIntegration> timeScheme_;
};

// Dimensional analysis utilities
namespace dimensions {
    
    // SI base dimensions [M L T Θ I]
    struct Dimension {
        int mass;         // kg
        int length;       // m
        int time;         // s
        int temperature;  // K
        int current;      // A
        
        // Check if dimensionless
        bool isDimensionless() const {
            return mass == 0 && length == 0 && time == 0 && 
                   temperature == 0 && current == 0;
        }
        
        // Arithmetic operations
        Dimension operator*(const Dimension& other) const;
        Dimension operator/(const Dimension& other) const;
        Dimension pow(int n) const;
    };
    
    // Common dimensions
    const Dimension DIMENSIONLESS{0, 0, 0, 0, 0};
    const Dimension LENGTH{0, 1, 0, 0, 0};
    const Dimension TIME{0, 0, 1, 0, 0};
    const Dimension MASS{1, 0, 0, 0, 0};
    const Dimension VELOCITY{0, 1, -1, 0, 0};
    const Dimension ACCELERATION{0, 1, -2, 0, 0};
    const Dimension FORCE{1, 1, -2, 0, 0};
    const Dimension PRESSURE{1, -1, -2, 0, 0};
    const Dimension ENERGY{1, 2, -2, 0, 0};
    const Dimension POWER{1, 2, -3, 0, 0};
    const Dimension DENSITY{1, -3, 0, 0, 0};
    const Dimension DYNAMIC_VISCOSITY{1, -1, -1, 0, 0};
    const Dimension KINEMATIC_VISCOSITY{0, 2, -1, 0, 0};
}

// Physical constants
namespace constants {
    constexpr Real R_universal = 8.314462618;      // J/(mol·K)
    constexpr Real k_Boltzmann = 1.380649e-23;     // J/K
    constexpr Real N_Avogadro = 6.02214076e23;     // 1/mol
    constexpr Real g_standard = 9.80665;           // m/s²
    constexpr Real sigma_StefanBoltzmann = 5.670374419e-8;  // W/(m²·K⁴)
}

// Non-dimensional numbers
class DimensionlessNumbers {
public:
    // Reynolds number: Re = ρUL/μ
    static Real Reynolds(Real density, Real velocity, Real length, Real viscosity) {
        return density * velocity * length / viscosity;
    }
    
    // Mach number: Ma = U/c
    static Real Mach(Real velocity, Real soundSpeed) {
        return velocity / soundSpeed;
    }
    
    // Prandtl number: Pr = μCp/k
    static Real Prandtl(Real viscosity, Real specificHeat, Real thermalConductivity) {
        return viscosity * specificHeat / thermalConductivity;
    }
    
    // Nusselt number: Nu = hL/k
    static Real Nusselt(Real heatTransferCoeff, Real length, Real thermalConductivity) {
        return heatTransferCoeff * length / thermalConductivity;
    }
    
    // Grashof number: Gr = gβΔTL³/ν²
    static Real Grashof(Real gravity, Real thermalExpansion, Real deltaT, 
                       Real length, Real kinematicViscosity) {
        return gravity * thermalExpansion * deltaT * cube(length) / sqr(kinematicViscosity);
    }
    
    // Rayleigh number: Ra = Gr·Pr
    static Real Rayleigh(Real grashof, Real prandtl) {
        return grashof * prandtl;
    }
    
    // Peclet number: Pe = Re·Pr = UL/α
    static Real Peclet(Real velocity, Real length, Real thermalDiffusivity) {
        return velocity * length / thermalDiffusivity;
    }
    
    // Strouhal number: St = fL/U
    static Real Strouhal(Real frequency, Real length, Real velocity) {
        return frequency * length / velocity;
    }
    
    // Froude number: Fr = U/√(gL)
    static Real Froude(Real velocity, Real gravity, Real length) {
        return velocity / std::sqrt(gravity * length);
    }
    
    // Weber number: We = ρU²L/σ
    static Real Weber(Real density, Real velocity, Real length, Real surfaceTension) {
        return density * sqr(velocity) * length / surfaceTension;
    }
    
    // Courant number: Co = UΔt/Δx
    static Real Courant(Real velocity, Real timeStep, Real cellSize) {
        return velocity * timeStep / cellSize;
    }
};

} // namespace cfd::physics