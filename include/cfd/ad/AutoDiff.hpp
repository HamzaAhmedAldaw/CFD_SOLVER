#pragma once

#include "cfd/ad/DualNumber.hpp"
#include "cfd/ad/TapeRecorder.hpp"
#include "cfd/core/Types.hpp"
#include <functional>
#include <vector>

namespace cfd::ad {

// Forward-mode automatic differentiation
class ForwardAD {
public:
    // Compute gradient of scalar function
    template<typename Func>
    static VectorX gradient(Func f, const VectorX& x) {
        const int n = x.size();
        VectorX grad(n);
        
        // Evaluate derivative w.r.t. each variable
        for (int i = 0; i < n; ++i) {
            // Create dual numbers with seed for i-th variable
            std::vector<DualReal> x_dual(n);
            for (int j = 0; j < n; ++j) {
                x_dual[j] = (i == j) ? DualReal(x[j], 0) : DualReal(x[j]);
            }
            
            // Evaluate function
            DualReal y = f(x_dual);
            grad[i] = y.derivative();
        }
        
        return grad;
    }
    
    // Compute Jacobian of vector function
    template<typename Func>
    static MatrixX jacobian(Func f, const VectorX& x) {
        const int n = x.size();
        
        // Evaluate with one dual number to get output dimension
        std::vector<DualReal> x_dual(n);
        for (int j = 0; j < n; ++j) {
            x_dual[j] = DualReal(x[j]);
        }
        auto y_test = f(x_dual);
        const int m = y_test.size();
        
        MatrixX jac(m, n);
        
        // Compute each column of Jacobian
        for (int i = 0; i < n; ++i) {
            // Seed i-th variable
            for (int j = 0; j < n; ++j) {
                x_dual[j] = (i == j) ? DualReal(x[j], 0) : DualReal(x[j]);
            }
            
            // Evaluate function
            auto y = f(x_dual);
            
            // Extract derivatives
            for (int j = 0; j < m; ++j) {
                jac(j, i) = y[j].derivative();
            }
        }
        
        return jac;
    }
    
    // Compute directional derivative
    template<typename Func>
    static Real directionalDerivative(Func f, const VectorX& x, const VectorX& v) {
        const int n = x.size();
        
        // Create dual numbers with direction v
        std::vector<DualReal> x_dual(n);
        for (int i = 0; i < n; ++i) {
            x_dual[i] = DualReal(x[i], v[i]);
        }
        
        // Evaluate function
        DualReal y = f(x_dual);
        return y.derivative();
    }
};

// Reverse-mode automatic differentiation (backpropagation)
class ReverseAD {
public:
    using Tape = TapeRecorder;
    
    // Compute gradient using tape
    template<typename Func>
    static VectorX gradient(Func f, const VectorX& x) {
        Tape tape;
        
        // Forward pass with recording
        std::vector<Tape::Variable> x_vars(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x_vars[i] = tape.newVariable(x[i]);
        }
        
        // Evaluate function
        Tape::Variable y = f(tape, x_vars);
        
        // Backward pass
        tape.gradient(y, x_vars);
        
        // Extract gradients
        VectorX grad(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            grad[i] = tape.derivative(y, x_vars[i]);
        }
        
        return grad;
    }
    
    // Compute Jacobian using reverse mode
    template<typename Func>
    static MatrixX jacobian(Func f, const VectorX& x) {
        // For Jacobian, reverse mode is efficient when m < n
        // Otherwise, use forward mode
        
        Tape tape;
        
        // Forward pass
        std::vector<Tape::Variable> x_vars(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x_vars[i] = tape.newVariable(x[i]);
        }
        
        // Evaluate function
        auto y_vars = f(tape, x_vars);
        const int m = y_vars.size();
        const int n = x.size();
        
        MatrixX jac(m, n);
        
        // Compute each row of Jacobian
        for (int i = 0; i < m; ++i) {
            tape.gradient(y_vars[i], x_vars);
            
            for (int j = 0; j < n; ++j) {
                jac(i, j) = tape.derivative(y_vars[i], x_vars[j]);
            }
            
            tape.reset();  // Clear for next row
        }
        
        return jac;
    }
};

// Automatic differentiation utilities
class ADUtils {
public:
    // Finite difference gradient (for verification)
    template<typename Func>
    static VectorX finiteDifferenceGradient(Func f, const VectorX& x, Real h = 1e-8) {
        const int n = x.size();
        VectorX grad(n);
        VectorX x_plus = x;
        VectorX x_minus = x;
        
        for (int i = 0; i < n; ++i) {
            x_plus[i] = x[i] + h;
            x_minus[i] = x[i] - h;
            
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h);
            
            x_plus[i] = x[i];
            x_minus[i] = x[i];
        }
        
        return grad;
    }
    
    // Check gradient implementation
    template<typename Func, typename GradFunc>
    static bool checkGradient(Func f, GradFunc grad_f, const VectorX& x,
                             Real tolerance = 1e-6) {
        VectorX ad_grad = ForwardAD::gradient(f, x);
        VectorX user_grad = grad_f(x);
        VectorX fd_grad = finiteDifferenceGradient(f, x);
        
        Real error_ad = (ad_grad - user_grad).norm();
        Real error_fd = (fd_grad - user_grad).norm();
        
        return error_ad < tolerance && error_fd < tolerance * 100;
    }
    
    // Hessian computation using AD
    template<typename Func>
    static MatrixX hessian(Func f, const VectorX& x) {
        const int n = x.size();
        MatrixX H(n, n);
        
        // Use forward-over-forward AD
        auto grad_f = [&f, n](const std::vector<DualReal>& x_dual) -> std::vector<DualReal> {
            return ForwardAD::gradient(f, x_dual);
        };
        
        // Compute Hessian columns
        for (int i = 0; i < n; ++i) {
            std::vector<DualNumber<DualReal, 1>> x_dual2(n);
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    x_dual2[j] = DualNumber<DualReal, 1>(DualReal(x[j], 0), 0);
                } else {
                    x_dual2[j] = DualNumber<DualReal, 1>(DualReal(x[j]));
                }
            }
            
            auto grad = grad_f(x_dual2);
            
            for (int j = 0; j < n; ++j) {
                H(j, i) = grad[j].derivative().derivative();
            }
        }
        
        // Ensure symmetry
        H = 0.5 * (H + H.transpose());
        
        return H;
    }
};

// Differentiation of field operations
template<typename T>
class FieldAD {
public:
    using DualField = Field<DualNumber<T>>;
    
    // Convert regular field to dual field
    static SharedPtr<DualField> toDualField(const Field<T>& field, int seedIndex = -1) {
        auto dualField = std::make_shared<DualField>(
            field.mesh(), field.name() + "_dual", field.fieldType());
        
        for (Index i = 0; i < field.mesh().numCells(); ++i) {
            if (seedIndex >= 0 && i == seedIndex) {
                (*dualField)[i] = DualNumber<T>(field[i], 0);
            } else {
                (*dualField)[i] = DualNumber<T>(field[i]);
            }
        }
        
        return dualField;
    }
    
    // Compute Jacobian of field residual
    template<typename ResidualFunc>
    static void computeJacobian(const Field<T>& field,
                               ResidualFunc residualFunc,
                               SparseMatrix& jacobian) {
        const Index n = field.mesh().numCells();
        std::vector<Triplet> triplets;
        triplets.reserve(n * 7);  // Estimate
        
        // For each cell, compute derivatives w.r.t. neighbors
        for (Index i = 0; i < n; ++i) {
            // Create dual field with seed at cell i
            auto dualField = toDualField(field, i);
            
            // Compute residual
            DualField residual(field.mesh(), "residual", field.fieldType());
            residualFunc(*dualField, residual);
            
            // Extract derivatives
            for (Index j = 0; j < n; ++j) {
                T deriv;
                if constexpr (std::is_same_v<T, Real>) {
                    deriv = residual[j].derivative();
                } else if constexpr (std::is_same_v<T, Vector3>) {
                    // Handle vector fields
                    for (int comp = 0; comp < 3; ++comp) {
                        deriv[comp] = residual[j].derivative()[comp];
                    }
                }
                
                if (std::abs(deriv) > EPSILON) {
                    triplets.emplace_back(j, i, deriv);
                }
            }
        }
        
        jacobian.setFromTriplets(triplets.begin(), triplets.end());
    }
};

// Adjoint method for optimization
template<typename StateType, typename ObjectiveType>
class AdjointMethod {
public:
    using StateFunc = std::function<void(const VectorX&, VectorX&)>;
    using ObjectiveFunc = std::function<ObjectiveType(const VectorX&)>;
    
    // Compute sensitivity of objective w.r.t. parameters
    static VectorX sensitivity(StateFunc stateEquation,
                              ObjectiveFunc objective,
                              const VectorX& parameters,
                              const VectorX& state) {
        // Solve adjoint equation: (∂R/∂u)^T λ = -∂J/∂u
        
        // Compute state equation Jacobian
        MatrixX dRdu = ForwardAD::jacobian(
            [&](const auto& u) { 
                VectorX R(u.size());
                stateEquation(u, R);
                return R;
            }, state);
        
        // Compute objective gradient w.r.t. state
        VectorX dJdu = ForwardAD::gradient(objective, state);
        
        // Solve adjoint equation
        VectorX lambda = dRdu.transpose().fullPivLu().solve(-dJdu);
        
        // Compute total derivative: dJ/dp = ∂J/∂p + λ^T ∂R/∂p
        VectorX dJdp = ForwardAD::gradient(
            [&](const auto& p) { return objective(state); }, parameters);
        
        MatrixX dRdp = ForwardAD::jacobian(
            [&](const auto& p) {
                VectorX R(state.size());
                stateEquation(state, R);
                return R;
            }, parameters);
        
        return dJdp + lambda.transpose() * dRdp;
    }
};

} // namespace cfd::ad