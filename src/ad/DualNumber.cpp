// === src/ad/DualNumber.cpp ===
#include "cfd/ad/DualNumber.hpp"

namespace cfd::ad {

// Explicit instantiation of common dual number types
template class DualNumber<Real, 1>;
template class DualNumber<Real, 2>;
template class DualNumber<Real, 3>;
template class DualNumber<Vector3, 1>;

// Specialized implementations for performance
template<>
DualNumber<Real, 1> pow(const DualNumber<Real, 1>& x, const DualNumber<Real, 1>& y) {
    // Optimized implementation for scalar dual numbers
    Real val = std::pow(x.value(), y.value());
    Real logx = std::log(x.value());
    Real deriv = val * (y.derivative(0) * logx + 
                        y.value() * x.derivative(0) / x.value());
    return DualNumber<Real, 1>(val, {deriv});
}

// Helper functions for dual number operations
namespace {
    // Ensure numerical stability in division
    template<typename T>
    T safeDivide(const T& num, const T& den) {
        if (std::abs(den) < SMALL) {
            return sign(num) * sign(den) * LARGE;
        }
        return num / den;
    }
}

} // namespace cfd::ad
