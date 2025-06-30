#pragma once

#include "cfd/physics/Physics.hpp"
#include "cfd/physics/Thermodynamics.hpp"
#include "cfd/physics/TurbulenceModel.hpp"
#include "cfd/solvers/PressureVelocityCoupling.hpp"

namespace cfd::physics {

// Navier-Stokes equations solver
class NavierStokes : public PhysicsModel {
public:
    struct Settings {
        // Flow type
        bool compressible = false;
        bool turbulent = false;
        bool buoyant = false;
        bool multiphase = false;
        
        // Physical properties
        Real density = 1.0;           // kg/m³
        Real viscosity = 1e-5;        // Pa·s
        Real specificHeatCp = 1000.0; // J/(kg·K)
        Real thermalConductivity = 0.025; // W/(m·K)
        
        // Reference values
        Real referencePressure = 101325.0;  // Pa
        Real referenceTemperature = 293.15;  // K
        Vector3 gravity = Vector3(0, 0, -9.81); // m/s²
        
        // Dimensionless parameters
        Real Reynolds = 100.0;
        Real Prandtl = 0.7;
        Real Mach = 0.1;
        
        // Solver settings
        SolverType solverType = SolverType::INCOMPRESSIBLE;
        std::string pressureVelocityCoupling = "SIMPLE";
        std::string turbulenceModel = "kOmegaSST";
    };
    
    NavierStokes(const Mesh& mesh, const Settings& settings = Settings());
    
    std::string name() const override { return "Navier-Stokes"; }
    void initialize() override;
    void update() override;
    void solve() override;
    Real maxTimeStep() const override;
    
    // Access to fields
    VectorField& velocity() { return *U_; }
    ScalarField& pressure() { return *p_; }
    ScalarField& temperature() { return *T_; }
    ScalarField& density() { return *rho_; }
    
    // Set boundary conditions
    void setVelocityBC(const std::string& patch, SharedPtr<BoundaryCondition<Vector3>> bc);
    void setPressureBC(const std::string& patch, SharedPtr<BoundaryCondition<Real>> bc);
    void setTemperatureBC(const std::string& patch, SharedPtr<BoundaryCondition<Real>> bc);
    
private:
    Settings settings_;
    
    // Primary fields
    SharedPtr<VectorField> U_;      // Velocity
    SharedPtr<ScalarField> p_;      // Pressure
    SharedPtr<ScalarField> T_;      // Temperature
    SharedPtr<ScalarField> rho_;    // Density
    
    // Material properties
    SharedPtr<ScalarField> mu_;     // Dynamic viscosity
    SharedPtr<ScalarField> nu_;     // Kinematic viscosity
    SharedPtr<ScalarField> lambda_; // Thermal conductivity
    SharedPtr<ScalarField> Cp_;     // Specific heat
    
    // Derived fields
    SharedPtr<VectorField> F_;      // Body forces
    SharedPtr<TensorField> tau_;    // Viscous stress tensor
    SharedPtr<VectorField> q_;      // Heat flux
    
    // Models
    SharedPtr<Thermodynamics> thermo_;
    SharedPtr<TurbulenceModel> turbulence_;
    SharedPtr<solvers::PressureVelocityCoupling> pvCoupling_;
    
    // Transport equations
    SharedPtr<TransportEquation<Vector3>> momentumEqn_;
    SharedPtr<TransportEquation<Real>> energyEqn_;
    SharedPtr<TransportEquation<Real>> continuityEqn_;
    
    // Solver functions
    void solveIncompressible();
    void solveCompressible();
    void solveLowMach();
    
    // Equation assembly
    void assembleMomentumEquation();
    void assembleEnergyEquation();
    void assembleContinuityEquation();
    
    // Source terms
    void addBuoyancy(VectorField& source);
    void addCoriolis(VectorField& source);
    void addPorosity(VectorField& source);
    
    // Stress tensor computation
    void computeStressTensor();
    void computeViscousForces();
    
    // Heat transfer
    void computeHeatFlux();
    void addViscousDissipation(ScalarField& source);
    
    // Time step calculation
    Real computeConvectiveTimeStep() const;
    Real computeDiffusiveTimeStep() const;
    Real computeAcousticTimeStep() const;
};

// Incompressible Navier-Stokes
class IncompressibleNS : public NavierStokes {
public:
    IncompressibleNS(const Mesh& mesh, const Settings& settings = Settings())
        : NavierStokes(mesh, settings) {
        settings_.compressible = false;
        settings_.solverType = SolverType::INCOMPRESSIBLE;
    }
    
    void solve() override;
    
private:
    // Specialized incompressible solver
    void projectVelocity();
    void solvePressurePoisson();
};

// Compressible Navier-Stokes
class CompressibleNS : public NavierStokes {
public:
    CompressibleNS(const Mesh& mesh, const Settings& settings = Settings())
        : NavierStokes(mesh, settings) {
        settings_.compressible = true;
        settings_.solverType = SolverType::COMPRESSIBLE;
    }
    
    void solve() override;
    
    // Additional compressible fields
    ScalarField& totalEnergy() { return *rhoE_; }
    ScalarField& soundSpeed() { return *c_; }
    
private:
    SharedPtr<ScalarField> rhoE_;   // Total energy
    SharedPtr<ScalarField> c_;      // Sound speed
    SharedPtr<ScalarField> Ma_;     // Local Mach number
    
    // Conservative variables
    SharedPtr<Field<CompressibleState>> conservativeState_;
    
    // Compressible flux computation
    void computeConvectiveFluxes();
    void computeViscousFluxes();
    
    // Shock capturing
    void addArtificialViscosity();
    Real computeShockSensor(Index cellId) const;
};

// Low-Mach number approximation
class LowMachNS : public NavierStokes {
public:
    LowMachNS(const Mesh& mesh, const Settings& settings = Settings())
        : NavierStokes(mesh, settings) {
        settings_.solverType = SolverType::LOW_MACH;
    }
    
    void solve() override;
    
private:
    // Thermodynamic pressure (spatially uniform)
    Real p0_;
    
    // Variable density effects
    void updateDensityFromTemperature();
    void solvePressureCorrection();
};

// Boussinesq approximation for buoyant flows
class BoussinesqNS : public IncompressibleNS {
public:
    struct BoussinesqSettings : Settings {
        Real referenceDensity = 1.0;
        Real thermalExpansion = 1e-3;  // 1/K
        Real referenceTemperature = 300.0;
    };
    
    BoussinesqNS(const Mesh& mesh, const BoussinesqSettings& settings)
        : IncompressibleNS(mesh, settings), boussinesqSettings_(settings) {
        settings_.buoyant = true;
    }
    
    void solve() override;
    
private:
    BoussinesqSettings boussinesqSettings_;
    
    // Buoyancy force: F = -ρ₀β(T-T₀)g
    void computeBuoyancyForce(VectorField& F);
};

// Non-Newtonian fluid models
class NonNewtonianNS : public IncompressibleNS {
public:
    enum Model {
        POWER_LAW,
        BINGHAM,
        HERSCHEL_BULKLEY,
        CARREAU,
        CROSS
    };
    
    NonNewtonianNS(const Mesh& mesh, Model model, const Settings& settings = Settings())
        : IncompressibleNS(mesh, settings), model_(model) {}
    
    void update() override;
    
private:
    Model model_;
    
    // Model parameters
    Real K_ = 1.0;      // Consistency index
    Real n_ = 1.0;      // Power law index
    Real tau0_ = 0.0;   // Yield stress
    Real mu0_ = 1.0;    // Zero shear viscosity
    Real muInf_ = 0.0;  // Infinite shear viscosity
    Real lambda_ = 1.0; // Time constant
    
    // Update viscosity based on strain rate
    void updateViscosity();
    Real computeApparentViscosity(Real strainRate) const;
    Real computeStrainRate(Index cellId) const;
};

// Multiphase Navier-Stokes (Volume of Fluid)
class MultiphaseNS : public NavierStokes {
public:
    struct Phase {
        std::string name;
        Real density;
        Real viscosity;
        Real surfaceTension;
    };
    
    MultiphaseNS(const Mesh& mesh, const std::vector<Phase>& phases,
                const Settings& settings = Settings())
        : NavierStokes(mesh, settings), phases_(phases) {
        settings_.multiphase = true;
    }
    
    void initialize() override;
    void solve() override;
    
    // Access phase fraction
    ScalarField& phaseFraction(int phase) { return *alpha_[phase]; }
    
private:
    std::vector<Phase> phases_;
    std::vector<SharedPtr<ScalarField>> alpha_;  // Phase fractions
    
    // Mixture properties
    void updateMixtureProperties();
    
    // Surface tension force
    void computeSurfaceTension(VectorField& F);
    
    // Phase advection
    void advectPhases();
    void compressiveFlux(ScalarField& alpha);
};

// Factory function
inline SharedPtr<NavierStokes> createNavierStokesSolver(
    const std::string& type,
    const Mesh& mesh,
    const NavierStokes::Settings& settings = NavierStokes::Settings()) {
    
    if (type == "incompressible") {
        return std::make_shared<IncompressibleNS>(mesh, settings);
    } else if (type == "compressible") {
        return std::make_shared<CompressibleNS>(mesh, settings);
    } else if (type == "lowMach") {
        return std::make_shared<LowMachNS>(mesh, settings);
    } else if (type == "boussinesq") {
        return std::make_shared<BoussinesqNS>(mesh, 
            BoussinesqNS::BoussinesqSettings());
    } else {
        throw std::runtime_error("Unknown Navier-Stokes solver type: " + type);
    }
}

} // namespace cfd::physics