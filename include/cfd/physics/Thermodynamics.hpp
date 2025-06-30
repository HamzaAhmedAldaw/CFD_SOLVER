#pragma once

#include "cfd/physics/Physics.hpp"
#include <map>
#include <string>

namespace cfd::physics {

// Thermodynamic model base class
class Thermodynamics {
public:
    Thermodynamics() = default;
    virtual ~Thermodynamics() = default;
    
    // Equation of state
    virtual Real density(Real pressure, Real temperature) const = 0;
    virtual Real pressure(Real density, Real temperature) const = 0;
    virtual Real temperature(Real pressure, Real density) const = 0;
    
    // Thermodynamic properties
    virtual Real specificHeatCp(Real temperature, Real pressure = 101325) const = 0;
    virtual Real specificHeatCv(Real temperature, Real pressure = 101325) const = 0;
    virtual Real enthalpy(Real temperature, Real pressure = 101325) const = 0;
    virtual Real entropy(Real temperature, Real pressure) const = 0;
    virtual Real internalEnergy(Real temperature) const = 0;
    
    // Transport properties
    virtual Real dynamicViscosity(Real temperature, Real pressure = 101325) const = 0;
    virtual Real thermalConductivity(Real temperature, Real pressure = 101325) const = 0;
    virtual Real prandtlNumber(Real temperature, Real pressure = 101325) const;
    
    // Derived properties
    virtual Real soundSpeed(Real temperature, Real pressure = 101325) const = 0;
    virtual Real specificGasConstant() const = 0;
    virtual Real gamma(Real temperature = 300) const { return specificHeatCp(temperature) / specificHeatCv(temperature); }
    
    // Field operations
    void updateDensity(ScalarField& rho, const ScalarField& p, const ScalarField& T) const;
    void updateTemperature(ScalarField& T, const ScalarField& p, const ScalarField& rho) const;
    void updateProperties(ScalarField& mu, ScalarField& lambda, const ScalarField& T) const;
};

// Perfect gas model
class PerfectGas : public Thermodynamics {
public:
    PerfectGas(Real R = 287.0, Real gamma = 1.4)
        : R_(R), gamma_(gamma) {
        Cp_ = gamma * R / (gamma - 1);
        Cv_ = R / (gamma - 1);
    }
    
    // Equation of state: p = ρRT
    Real density(Real pressure, Real temperature) const override {
        return pressure / (R_ * temperature);
    }
    
    Real pressure(Real density, Real temperature) const override {
        return density * R_ * temperature;
    }
    
    Real temperature(Real pressure, Real density) const override {
        return pressure / (density * R_);
    }
    
    // Thermodynamic properties
    Real specificHeatCp(Real /*temperature*/, Real /*pressure*/) const override { return Cp_; }
    Real specificHeatCv(Real /*temperature*/, Real /*pressure*/) const override { return Cv_; }
    Real enthalpy(Real temperature, Real /*pressure*/) const override { return Cp_ * temperature; }
    Real internalEnergy(Real temperature) const override { return Cv_ * temperature; }
    Real entropy(Real temperature, Real pressure) const override;
    
    // Transport properties (Sutherland's law)
    Real dynamicViscosity(Real temperature, Real /*pressure*/) const override;
    Real thermalConductivity(Real temperature, Real /*pressure*/) const override;
    
    // Derived properties
    Real soundSpeed(Real temperature, Real /*pressure*/) const override {
        return std::sqrt(gamma_ * R_ * temperature);
    }
    
    Real specificGasConstant() const override { return R_; }
    Real gamma(Real /*temperature*/) const override { return gamma_; }
    
private:
    Real R_;      // Specific gas constant
    Real gamma_;  // Heat capacity ratio
    Real Cp_;     // Specific heat at constant pressure
    Real Cv_;     // Specific heat at constant volume
    
    // Sutherland's law parameters
    Real mu0_ = 1.716e-5;   // Reference viscosity
    Real T0_ = 273.15;      // Reference temperature
    Real S_ = 110.4;        // Sutherland temperature
};

// Real gas models
class RealGas : public Thermodynamics {
public:
    // Van der Waals equation of state
    class VanDerWaals : public RealGas {
    public:
        VanDerWaals(Real a, Real b, Real R, Real M)
            : a_(a), b_(b), R_(R), M_(M) {}
        
        Real pressure(Real density, Real temperature) const override;
        Real specificGasConstant() const override { return R_ / M_; }
        
    private:
        Real a_, b_;  // Van der Waals constants
        Real R_;      // Universal gas constant
        Real M_;      // Molar mass
    };
    
    // Peng-Robinson equation of state
    class PengRobinson : public RealGas {
    public:
        PengRobinson(Real Tc, Real pc, Real omega, Real M);
        
        Real pressure(Real density, Real temperature) const override;
        Real fugacityCoefficient(Real pressure, Real temperature) const;
        
    private:
        Real Tc_;     // Critical temperature
        Real pc_;     // Critical pressure
        Real omega_;  // Acentric factor
        Real M_;      // Molar mass
        Real a_, b_;  // PR constants
    };
};

// Incompressible fluid model
class IncompressibleFluid : public Thermodynamics {
public:
    IncompressibleFluid(Real rho0 = 1000.0, Real Cp = 4186.0,
                       Real mu = 1e-3, Real lambda = 0.6)
        : rho0_(rho0), Cp_(Cp), mu_(mu), lambda_(lambda) {}
    
    // Constant density
    Real density(Real /*pressure*/, Real /*temperature*/) const override { return rho0_; }
    Real pressure(Real /*density*/, Real /*temperature*/) const override { return 0; }
    Real temperature(Real /*pressure*/, Real /*density*/) const override { return 300; }
    
    // Properties
    Real specificHeatCp(Real /*T*/, Real /*p*/) const override { return Cp_; }
    Real specificHeatCv(Real /*T*/, Real /*p*/) const override { return Cp_; }
    Real dynamicViscosity(Real /*T*/, Real /*p*/) const override { return mu_; }
    Real thermalConductivity(Real /*T*/, Real /*p*/) const override { return lambda_; }
    
    Real enthalpy(Real temperature, Real /*pressure*/) const override { return Cp_ * temperature; }
    Real internalEnergy(Real temperature) const override { return Cp_ * temperature; }
    Real entropy(Real /*T*/, Real /*p*/) const override { return 0; }
    Real soundSpeed(Real /*T*/, Real /*p*/) const override { return 1e10; } // Large value
    Real specificGasConstant() const override { return 0; }
    
private:
    Real rho0_;    // Constant density
    Real Cp_;      // Specific heat
    Real mu_;      // Dynamic viscosity
    Real lambda_;  // Thermal conductivity
};

// Polynomial thermodynamic properties (NASA polynomials)
class PolynomialThermo : public PerfectGas {
public:
    struct Coefficients {
        Real Tlow, Thigh, Tcommon;
        std::array<Real, 7> lowT;   // Coefficients for T < Tcommon
        std::array<Real, 7> highT;  // Coefficients for T > Tcommon
    };
    
    PolynomialThermo(const Coefficients& coeffs, Real R)
        : PerfectGas(R), coeffs_(coeffs) {}
    
    Real specificHeatCp(Real T, Real /*p*/) const override;
    Real enthalpy(Real T, Real /*p*/) const override;
    Real entropy(Real T, Real p) const override;
    
private:
    Coefficients coeffs_;
    
    const std::array<Real, 7>& selectCoeffs(Real T) const {
        return T < coeffs_.Tcommon ? coeffs_.lowT : coeffs_.highT;
    }
};

// Mixture thermodynamics
class MixtureThermo : public Thermodynamics {
public:
    struct Species {
        std::string name;
        Real molarMass;
        SharedPtr<Thermodynamics> thermo;
    };
    
    MixtureThermo() = default;
    
    // Add species
    void addSpecies(const Species& species) {
        species_[species.name] = species;
    }
    
    // Set composition (mass fractions)
    void setComposition(const std::map<std::string, Real>& massFractions);
    
    // Mixture properties
    Real density(Real pressure, Real temperature) const override;
    Real pressure(Real density, Real temperature) const override;
    Real temperature(Real pressure, Real density) const override;
    
    Real specificHeatCp(Real T, Real p) const override;
    Real specificHeatCv(Real T, Real p) const override;
    Real enthalpy(Real T, Real p) const override;
    Real entropy(Real T, Real p) const override;
    Real internalEnergy(Real T) const override;
    
    Real dynamicViscosity(Real T, Real p) const override;
    Real thermalConductivity(Real T, Real p) const override;
    Real soundSpeed(Real T, Real p) const override;
    Real specificGasConstant() const override;
    
    // Species properties
    Real speciesDensity(const std::string& species, Real p, Real T) const;
    Real speciesEnthalpy(const std::string& species, Real T) const;
    
private:
    std::map<std::string, Species> species_;
    std::map<std::string, Real> massFractions_;
    std::map<std::string, Real> moleFractions_;
    Real meanMolarMass_;
    
    void updateMoleFractions();
    
    // Mixing rules
    Real wilkeMixtureViscosity(Real T, Real p) const;
    Real wassiljewaThermalConductivity(Real T, Real p) const;
};

// Phase change thermodynamics
class PhaseChangeThermo : public Thermodynamics {
public:
    PhaseChangeThermo(SharedPtr<Thermodynamics> liquid,
                     SharedPtr<Thermodynamics> vapor,
                     Real Tsat, Real hfg)
        : liquid_(liquid), vapor_(vapor), Tsat_(Tsat), hfg_(hfg) {}
    
    // Phase determination
    enum Phase { LIQUID, VAPOR, MIXTURE };
    Phase determinePhase(Real temperature, Real quality) const;
    
    // Two-phase properties
    Real vaporQuality(Real enthalpy, Real temperature) const;
    Real saturationTemperature(Real pressure) const;
    Real saturationPressure(Real temperature) const;
    
    // Mixture properties based on quality
    Real density(Real pressure, Real temperature) const override;
    Real specificHeatCp(Real T, Real p) const override;
    Real enthalpy(Real T, Real p) const override;
    
private:
    SharedPtr<Thermodynamics> liquid_;
    SharedPtr<Thermodynamics> vapor_;
    Real Tsat_;  // Saturation temperature
    Real hfg_;   // Latent heat of vaporization
    
    // Clausius-Clapeyron relation
    Real claussiusClapeyron(Real T) const;
};

// Table-based thermodynamics
class TabulatedThermo : public Thermodynamics {
public:
    TabulatedThermo(const std::string& filename);
    
    // Interpolated properties
    Real density(Real pressure, Real temperature) const override;
    Real specificHeatCp(Real T, Real p) const override;
    Real enthalpy(Real T, Real p) const override;
    Real dynamicViscosity(Real T, Real p) const override;
    Real thermalConductivity(Real T, Real p) const override;
    Real soundSpeed(Real T, Real p) const override;
    Real specificGasConstant() const override { return R_; }
    
private:
    // Property tables
    std::vector<Real> pressureTable_;
    std::vector<Real> temperatureTable_;
    std::vector<std::vector<Real>> densityTable_;
    std::vector<std::vector<Real>> enthalpyTable_;
    std::vector<std::vector<Real>> cpTable_;
    std::vector<std::vector<Real>> viscosityTable_;
    std::vector<std::vector<Real>> conductivityTable_;
    Real R_;
    
    // 2D interpolation
    Real interpolate2D(const std::vector<std::vector<Real>>& table,
                      Real p, Real T) const;
    
    // Find table indices
    std::pair<int, int> findIndices(const std::vector<Real>& vec, Real val) const;
};

// Factory function
inline SharedPtr<Thermodynamics> createThermodynamicsModel(
    const std::string& type,
    const std::map<std::string, Real>& parameters = {}) {
    
    if (type == "perfectGas") {
        Real R = parameters.count("R") ? parameters.at("R") : 287.0;
        Real gamma = parameters.count("gamma") ? parameters.at("gamma") : 1.4;
        return std::make_shared<PerfectGas>(R, gamma);
    } else if (type == "incompressible") {
        Real rho = parameters.count("density") ? parameters.at("density") : 1000.0;
        Real Cp = parameters.count("Cp") ? parameters.at("Cp") : 4186.0;
        return std::make_shared<IncompressibleFluid>(rho, Cp);
    } else if (type == "mixture") {
        return std::make_shared<MixtureThermo>();
    } else {
        throw std::runtime_error("Unknown thermodynamics model: " + type);
    }
}

} // namespace cfd::physics