#pragma once

#include "cfd/physics/Physics.hpp"
#include "cfd/core/Field.hpp"
#include <memory>

namespace cfd::physics {

// Base turbulence model class
class TurbulenceModel : public PhysicsModel {
public:
    TurbulenceModel(const Mesh& mesh) : PhysicsModel(mesh) {}
    
    // Turbulent viscosity
    virtual ScalarField& nut() = 0;
    virtual const ScalarField& nut() const = 0;
    
    // Turbulent kinetic energy (if applicable)
    virtual ScalarField* k() { return nullptr; }
    virtual const ScalarField* k() const { return nullptr; }
    
    // Specific dissipation rate (if applicable)
    virtual ScalarField* omega() { return nullptr; }
    virtual const ScalarField* omega() const { return nullptr; }
    
    // Dissipation rate (if applicable)
    virtual ScalarField* epsilon() { return nullptr; }
    virtual const ScalarField* epsilon() const { return nullptr; }
    
    // Effective viscosity: μ_eff = μ + μ_t
    virtual void effectiveViscosity(ScalarField& muEff, 
                                   const ScalarField& mu) const {
        muEff = mu + nut();
    }
    
    // Turbulent Prandtl number
    virtual Real turbulentPrandtl() const { return 0.85; }
    
    // Wall functions
    virtual void applyWallFunctions() {}
    
    // y+ calculation
    virtual void calculateYPlus(ScalarField& yPlus,
                               const ScalarField& rho,
                               const ScalarField& mu) const;
    
protected:
    SharedPtr<ScalarField> nut_;  // Turbulent viscosity
    
    // Common turbulence calculations
    Real vonKarmanConstant_ = 0.41;
    Real Cmu_ = 0.09;
    Real E_ = 9.8;  // Wall function constant
    
    // Wall distance
    SharedPtr<ScalarField> wallDistance_;
    void computeWallDistance();
    
    // Strain rate magnitude
    Real strainRateMagnitude(Index cellId, const VectorField& U) const;
    Real vorticity(Index cellId, const VectorField& U) const;
};

// Reynolds-Averaged Navier-Stokes (RANS) models
class RANSModel : public TurbulenceModel {
public:
    RANSModel(const Mesh& mesh) : TurbulenceModel(mesh) {}
    
    // Reynolds stress tensor
    virtual void reynoldsStress(TensorField& R,
                               const VectorField& U) const;
    
    // Production term
    virtual Real production(Index cellId,
                           const VectorField& U,
                           const TensorField& gradU) const;
    
protected:
    // Model constants (can be overridden)
    Real sigmak_ = 1.0;
    Real sigmaEps_ = 1.3;
    Real C1_ = 1.44;
    Real C2_ = 1.92;
};

// Spalart-Allmaras model
class SpalartAllmaras : public RANSModel {
public:
    SpalartAllmaras(const Mesh& mesh);
    
    std::string name() const override { return "Spalart-Allmaras"; }
    void initialize() override;
    void update() override;
    void solve() override;
    
    ScalarField& nut() override { return *nut_; }
    const ScalarField& nut() const override { return *nut_; }
    
    // SA working variable
    ScalarField& nuTilde() { return *nuTilde_; }
    
private:
    SharedPtr<ScalarField> nuTilde_;  // SA variable
    
    // Model constants
    Real cb1_ = 0.1355;
    Real cb2_ = 0.622;
    Real cv1_ = 7.1;
    Real cv2_ = 5.0;
    Real cw1_;  // Derived from other constants
    Real cw2_ = 0.3;
    Real cw3_ = 2.0;
    Real sigma_ = 2.0/3.0;
    
    // Model functions
    Real fv1(Real chi) const;
    Real fv2(Real chi) const;
    Real fw(Real r) const;
    Real S_tilde(Index cellId, const VectorField& U) const;
    
    // Solve SA transport equation
    void solveNuTilde(const VectorField& U, const ScalarField& nu);
    void updateNut(const ScalarField& nu);
};

// k-epsilon model
class KEpsilon : public RANSModel {
public:
    KEpsilon(const Mesh& mesh);
    
    std::string name() const override { return "k-epsilon"; }
    void initialize() override;
    void update() override;
    void solve() override;
    
    ScalarField& nut() override { return *nut_; }
    const ScalarField& nut() const override { return *nut_; }
    ScalarField* k() override { return k_.get(); }
    const ScalarField* k() const override { return k_.get(); }
    ScalarField* epsilon() override { return epsilon_.get(); }
    const ScalarField* epsilon() const override { return epsilon_.get(); }
    
protected:
    SharedPtr<ScalarField> k_;        // Turbulent kinetic energy
    SharedPtr<ScalarField> epsilon_;  // Dissipation rate
    
    // Standard k-epsilon constants
    Real Cmu_ = 0.09;
    Real C1eps_ = 1.44;
    Real C2eps_ = 1.92;
    Real sigmak_ = 1.0;
    Real sigmaEps_ = 1.3;
    
    // Solve transport equations
    virtual void solveK(const VectorField& U, const ScalarField& nu);
    virtual void solveEpsilon(const VectorField& U, const ScalarField& nu);
    void updateNut();
    
    // Wall functions
    void epsilonWallFunction();
};

// Realizable k-epsilon model
class RealizableKEpsilon : public KEpsilon {
public:
    RealizableKEpsilon(const Mesh& mesh) : KEpsilon(mesh) {}
    
    std::string name() const override { return "realizable k-epsilon"; }
    void update() override;
    
protected:
    // Variable Cmu
    Real computeCmu(Index cellId, const VectorField& U) const;
    
    // Modified epsilon equation
    void solveEpsilon(const VectorField& U, const ScalarField& nu) override;
};

// k-omega model
class KOmega : public RANSModel {
public:
    KOmega(const Mesh& mesh);
    
    std::string name() const override { return "k-omega"; }
    void initialize() override;
    void update() override;
    void solve() override;
    
    ScalarField& nut() override { return *nut_; }
    const ScalarField& nut() const override { return *nut_; }
    ScalarField* k() override { return k_.get(); }
    const ScalarField* k() const override { return k_.get(); }
    ScalarField* omega() override { return omega_.get(); }
    const ScalarField* omega() const override { return omega_.get(); }
    
protected:
    SharedPtr<ScalarField> k_;      // Turbulent kinetic energy
    SharedPtr<ScalarField> omega_;  // Specific dissipation rate
    
    // Standard k-omega constants
    Real betaStar_ = 0.09;
    Real beta_ = 0.075;
    Real gamma_ = 5.0/9.0;
    Real sigmak_ = 2.0;
    Real sigmaOmega_ = 2.0;
    
    // Solve transport equations
    virtual void solveK(const VectorField& U, const ScalarField& nu);
    virtual void solveOmega(const VectorField& U, const ScalarField& nu);
    void updateNut();
};

// k-omega SST model
class KOmegaSST : public KOmega {
public:
    KOmegaSST(const Mesh& mesh);
    
    std::string name() const override { return "k-omega SST"; }
    void update() override;
    void solve() override;
    
protected:
    // SST blending functions
    SharedPtr<ScalarField> F1_;  // Blending function 1
    SharedPtr<ScalarField> F2_;  // Blending function 2
    
    // SST constants (two sets)
    struct Constants {
        Real beta, gamma, sigmak, sigmaOmega;
    };
    Constants inner_{0.075, 5.0/9.0, 0.85, 0.5};     // k-omega
    Constants outer_{0.0828, 0.44, 1.0, 0.856};     // k-epsilon
    
    Real a1_ = 0.31;
    
    // Blending functions
    void computeBlendingFunctions(const VectorField& U, const ScalarField& nu);
    Real blend(Real inner, Real outer, Real F1) const {
        return F1 * inner + (1.0 - F1) * outer;
    }
    
    // Cross-diffusion term
    Real crossDiffusion(Index cellId) const;
    
    // Limited eddy viscosity
    void updateNutSST(const VectorField& U);
};

// Large Eddy Simulation (LES) models
class LESModel : public TurbulenceModel {
public:
    LESModel(const Mesh& mesh) : TurbulenceModel(mesh) {}
    
    // Subgrid scale viscosity
    virtual void updateSGSViscosity(const VectorField& U) = 0;
    
    // Filter width
    virtual Real filterWidth(Index cellId) const;
    
    // Dynamic procedure (if applicable)
    virtual bool isDynamic() const { return false; }
    
protected:
    // Calculate filter width
    Real deltaFilter_;
    void computeFilterWidth();
};

// Smagorinsky model
class Smagorinsky : public LESModel {
public:
    Smagorinsky(const Mesh& mesh, Real Cs = 0.1)
        : LESModel(mesh), Cs_(Cs) {}
    
    std::string name() const override { return "Smagorinsky"; }
    void initialize() override;
    void update() override;
    void solve() override {}  // Algebraic model
    
    ScalarField& nut() override { return *nut_; }
    const ScalarField& nut() const override { return *nut_; }
    
    void updateSGSViscosity(const VectorField& U) override;
    
protected:
    Real Cs_;  // Smagorinsky constant
};

// Dynamic Smagorinsky model
class DynamicSmagorinsky : public Smagorinsky {
public:
    DynamicSmagorinsky(const Mesh& mesh)
        : Smagorinsky(mesh, 0.0) {}  // Cs computed dynamically
    
    std::string name() const override { return "Dynamic Smagorinsky"; }
    bool isDynamic() const override { return true; }
    
    void updateSGSViscosity(const VectorField& U) override;
    
private:
    SharedPtr<ScalarField> Cs_field_;  // Dynamic Cs field
    
    // Germano identity
    void computeDynamicCs(const VectorField& U);
    
    // Test filter
    void testFilter(const TensorField& S, TensorField& S_test) const;
};

// WALE (Wall-Adapting Local Eddy-viscosity) model
class WALE : public LESModel {
public:
    WALE(const Mesh& mesh, Real Cw = 0.325)
        : LESModel(mesh), Cw_(Cw) {}
    
    std::string name() const override { return "WALE"; }
    void initialize() override;
    void update() override;
    void solve() override {}  // Algebraic model
    
    ScalarField& nut() override { return *nut_; }
    const ScalarField& nut() const override { return *nut_; }
    
    void updateSGSViscosity(const VectorField& U) override;
    
private:
    Real Cw_;  // WALE constant
    
    // Velocity gradient tensor squared
    Matrix3 computeSd(const TensorField& gradU, Index cellId) const;
};

// Detached Eddy Simulation (DES) models
class DESModel : public TurbulenceModel {
public:
    DESModel(const Mesh& mesh, SharedPtr<RANSModel> ransModel)
        : TurbulenceModel(mesh), ransModel_(ransModel) {}
    
    // Switch between RANS and LES
    virtual Real lengthScale(Index cellId) const = 0;
    
protected:
    SharedPtr<RANSModel> ransModel_;
    SharedPtr<ScalarField> FDES_;  // DES blending function
    
    Real CDES_ = 0.65;  // DES constant
};

// Spalart-Allmaras DES
class SADES : public DESModel {
public:
    SADES(const Mesh& mesh);
    
    std::string name() const override { return "SA-DES"; }
    void initialize() override;
    void update() override;
    void solve() override;
    
    ScalarField& nut() override { return ransModel_->nut(); }
    const ScalarField& nut() const override { return ransModel_->nut(); }
    
    Real lengthScale(Index cellId) const override;
};

// SST-DES
class SSTDES : public DESModel {
public:
    SSTDES(const Mesh& mesh);
    
    std::string name() const override { return "SST-DES"; }
    void initialize() override;
    void update() override;
    void solve() override;
    
    ScalarField& nut() override { return ransModel_->nut(); }
    const ScalarField& nut() const override { return ransModel_->nut(); }
    
    Real lengthScale(Index cellId) const override;
};

// Factory function
inline SharedPtr<TurbulenceModel> createTurbulenceModel(
    TurbulenceModelType type,
    const Mesh& mesh) {
    
    switch (type) {
        case TurbulenceModelType::NONE:
            return nullptr;
        case TurbulenceModelType::SPALART_ALLMARAS:
            return std::make_shared<SpalartAllmaras>(mesh);
        case TurbulenceModelType::K_EPSILON:
            return std::make_shared<KEpsilon>(mesh);
        case TurbulenceModelType::K_OMEGA:
            return std::make_shared<KOmega>(mesh);
        case TurbulenceModelType::K_OMEGA_SST:
            return std::make_shared<KOmegaSST>(mesh);
        case TurbulenceModelType::LES_SMAGORINSKY:
            return std::make_shared<Smagorinsky>(mesh);
        case TurbulenceModelType::LES_WALE:
            return std::make_shared<WALE>(mesh);
        default:
            throw std::runtime_error("Unknown turbulence model type");
    }
}

} // namespace cfd::physics