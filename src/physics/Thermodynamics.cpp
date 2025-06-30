// === src/physics/Thermodynamics.cpp ===
#include "cfd/physics/Thermodynamics.hpp"
#include <cmath>
#include <stdexcept>

namespace cfd::physics {

Thermodynamics::Thermodynamics(EquationOfState eos,
                             const PhysicsParameters& params)
    : eos_(eos), params_(params) {
}

Real Thermodynamics::density(Real p, Real T) const {
    switch (eos_) {
        case EquationOfState::IDEAL_GAS:
            return p / (params_.R * T);
            
        case EquationOfState::INCOMPRESSIBLE:
            return params_.rho0;
            
        case EquationOfState::BOUSSINESQ:
            return params_.rho0 * (1 - params_.beta * (T - params_.T0));
            
        case EquationOfState::STIFFENED_GAS:
            return (p + params_.pInf) / (params_.R * T);
            
        case EquationOfState::VAN_DER_WAALS:
            // Solve cubic equation for density
            // Simplified implementation
            return p / (params_.R * T); // Fallback to ideal gas
            
        default:
            throw std::runtime_error("Unknown equation of state");
    }
}

Real Thermodynamics::pressure(Real rho, Real T) const {
    switch (eos_) {
        case EquationOfState::IDEAL_GAS:
            return rho * params_.R * T;
            
        case EquationOfState::INCOMPRESSIBLE:
            return 0.0; // Pressure is dynamic only
            
        case EquationOfState::BOUSSINESQ:
            return 0.0; // Pressure is dynamic only
            
        case EquationOfState::STIFFENED_GAS:
            return rho * params_.R * T - params_.pInf;
            
        case EquationOfState::VAN_DER_WAALS:
            Real v = 1.0 / rho; // Specific volume
            return params_.R * T / (v - params_.b) - params_.a / (v * v);
            
        default:
            throw std::runtime_error("Unknown equation of state");
    }
}

Real Thermodynamics::temperature(Real p, Real rho) const {
    switch (eos_) {
        case EquationOfState::IDEAL_GAS:
            return p / (rho * params_.R);
            
        case EquationOfState::INCOMPRESSIBLE:
            return params_.T0; // Isothermal
            
        case EquationOfState::BOUSSINESQ:
            return params_.T0 + (1 - params_.rho0/rho) / params_.beta;
            
        case EquationOfState::STIFFENED_GAS:
            return (p + params_.pInf) / (rho * params_.R);
            
        case EquationOfState::VAN_DER_WAALS:
            Real v = 1.0 / rho;
            return (p + params_.a / (v * v)) * (v - params_.b) / params_.R;
            
        default:
            throw std::runtime_error("Unknown equation of state");
    }
}

Real Thermodynamics::specificHeatCp(Real T) const {
    if (params_.cpModel == SpecificHeatModel::CONSTANT) {
        return params_.Cp0;
    } else if (params_.cpModel == SpecificHeatModel::POLYNOMIAL) {
        // NASA polynomial
        Real T2 = T * T;
        Real T3 = T2 * T;
        Real T4 = T3 * T;
        
        return params_.R * (params_.cpCoeffs[0] + 
                           params_.cpCoeffs[1] * T +
                           params_.cpCoeffs[2] * T2 +
                           params_.cpCoeffs[3] * T3 +
                           params_.cpCoeffs[4] * T4);
    }
    
    return params_.Cp0;
}

Real Thermodynamics::specificHeatCv(Real T) const {
    return specificHeatCp(T) - params_.R;
}

Real Thermodynamics::enthalpy(Real T) const {
    if (params_.cpModel == SpecificHeatModel::CONSTANT) {
        return params_.Cp0 * T;
    } else if (params_.cpModel == SpecificHeatModel::POLYNOMIAL) {
        Real T2 = T * T;
        Real T3 = T2 * T;
        Real T4 = T3 * T;
        Real T5 = T4 * T;
        
        return params_.R * T * (params_.cpCoeffs[0] + 
                               params_.cpCoeffs[1] * T / 2 +
                               params_.cpCoeffs[2] * T2 / 3 +
                               params_.cpCoeffs[3] * T3 / 4 +
                               params_.cpCoeffs[4] * T4 / 5 +
                               params_.cpCoeffs[5] / T);
    }
    
    return params_.Cp0 * T;
}

Real Thermodynamics::entropy(Real p, Real T) const {
    if (eos_ == EquationOfState::IDEAL_GAS) {
        Real p0 = params_.p0;
        Real T0 = params_.T0;
        
        if (params_.cpModel == SpecificHeatModel::CONSTANT) {
            return params_.Cp0 * std::log(T/T0) - params_.R * std::log(p/p0);
        }
    }
    
    // Simplified implementation
    return 0.0;
}

Real Thermodynamics::thermalConductivity(Real T) const {
    if (params_.thermalConductivityModel == ThermalConductivityModel::CONSTANT) {
        return params_.k0;
    } else if (params_.thermalConductivityModel == ThermalConductivityModel::SUTHERLAND) {
        // Sutherland's law for thermal conductivity
        Real T0 = params_.T0;
        Real S = 194.0; // Sutherland's constant for thermal conductivity
        return params_.k0 * std::pow(T/T0, 1.5) * (T0 + S) / (T + S);
    }
    
    return params_.k0;
}

Real Thermodynamics::soundSpeed(Real p, Real rho) const {
    switch (eos_) {
        case EquationOfState::IDEAL_GAS:
            return std::sqrt(params_.gamma * p / rho);
            
        case EquationOfState::INCOMPRESSIBLE:
            return 1e10; // Very large (infinite)
            
        case EquationOfState::STIFFENED_GAS:
            return std::sqrt(params_.gamma * (p + params_.pInf) / rho);
            
        default:
            // Use numerical derivative
            Real dp = 0.001 * p;
            Real rho1 = density(p - dp, temperature(p, rho));
            Real rho2 = density(p + dp, temperature(p, rho));
            Real dpdRho = 2 * dp / (rho2 - rho1);
            return std::sqrt(dpdRho);
    }
}
