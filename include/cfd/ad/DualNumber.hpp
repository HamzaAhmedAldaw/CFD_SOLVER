#pragma once

#include "cfd/core/Types.hpp"
#include <cmath>
#include <iostream>
#include <array>

namespace cfd::ad {

// Forward-mode automatic differentiation using dual numbers
template<typename T, int N = 1>
class DualNumber {
public:
    using value_type = T;
    static constexpr int num_derivatives = N;
    
    // Constructors
    DualNumber() : value_(T(0)) {
        derivatives_.fill(T(0));
    }
    
    explicit DualNumber(const T& value) : value_(value) {
        derivatives_.fill(T(0));
    }
    
    DualNumber(const T& value, int index) : value_(value) {
        derivatives_.fill(T(0));
        if (index >= 0 && index < N) {
            derivatives_[index] = T(1);
        }
    }
    
    DualNumber(const T& value, const std::array<T, N>& derivatives)
        : value_(value), derivatives_(derivatives) {}
    
    // Copy and assignment
    DualNumber(const DualNumber&) = default;
    DualNumber& operator=(const DualNumber&) = default;
    
    // Value and derivative access
    const T& value() const { return value_; }
    T& value() { return value_; }
    
    const T& derivative(int i = 0) const { return derivatives_[i]; }
    T& derivative(int i = 0) { return derivatives_[i]; }
    
    const std::array<T, N>& derivatives() const { return derivatives_; }
    std::array<T, N>& derivatives() { return derivatives_; }
    
    // Conversion to value type
    explicit operator T() const { return value_; }
    
    // Arithmetic operators
    DualNumber operator+() const { return *this; }
    
    DualNumber operator-() const {
        DualNumber result(-value_);
        for (int i = 0; i < N; ++i) {
            result.derivatives_[i] = -derivatives_[i];
        }
        return result;
    }
    
    DualNumber& operator+=(const DualNumber& rhs) {
        value_ += rhs.value_;
        for (int i = 0; i < N; ++i) {
            derivatives_[i] += rhs.derivatives_[i];
        }
        return *this;
    }
    
    DualNumber& operator-=(const DualNumber& rhs) {
        value_ -= rhs.value_;
        for (int i = 0; i < N; ++i) {
            derivatives_[i] -= rhs.derivatives_[i];
        }
        return *this;
    }
    
    DualNumber& operator*=(const DualNumber& rhs) {
        // Product rule: (fg)' = f'g + fg'
        for (int i = 0; i < N; ++i) {
            derivatives_[i] = derivatives_[i] * rhs.value_ + value_ * rhs.derivatives_[i];
        }
        value_ *= rhs.value_;
        return *this;
    }
    
    DualNumber& operator/=(const DualNumber& rhs) {
        // Quotient rule: (f/g)' = (f'g - fg')/g²
        T inv_g = T(1) / rhs.value_;
        T inv_g2 = inv_g * inv_g;
        
        for (int i = 0; i < N; ++i) {
            derivatives_[i] = (derivatives_[i] * rhs.value_ - value_ * rhs.derivatives_[i]) * inv_g2;
        }
        value_ *= inv_g;
        return *this;
    }
    
    // Scalar operations
    DualNumber& operator+=(const T& scalar) {
        value_ += scalar;
        return *this;
    }
    
    DualNumber& operator-=(const T& scalar) {
        value_ -= scalar;
        return *this;
    }
    
    DualNumber& operator*=(const T& scalar) {
        value_ *= scalar;
        for (int i = 0; i < N; ++i) {
            derivatives_[i] *= scalar;
        }
        return *this;
    }
    
    DualNumber& operator/=(const T& scalar) {
        T inv = T(1) / scalar;
        value_ *= inv;
        for (int i = 0; i < N; ++i) {
            derivatives_[i] *= inv;
        }
        return *this;
    }
    
private:
    T value_;
    std::array<T, N> derivatives_;
};

// Binary operators
template<typename T, int N>
DualNumber<T, N> operator+(const DualNumber<T, N>& lhs, const DualNumber<T, N>& rhs) {
    DualNumber<T, N> result = lhs;
    result += rhs;
    return result;
}

template<typename T, int N>
DualNumber<T, N> operator-(const DualNumber<T, N>& lhs, const DualNumber<T, N>& rhs) {
    DualNumber<T, N> result = lhs;
    result -= rhs;
    return result;
}

template<typename T, int N>
DualNumber<T, N> operator*(const DualNumber<T, N>& lhs, const DualNumber<T, N>& rhs) {
    DualNumber<T, N> result = lhs;
    result *= rhs;
    return result;
}

template<typename T, int N>
DualNumber<T, N> operator/(const DualNumber<T, N>& lhs, const DualNumber<T, N>& rhs) {
    DualNumber<T, N> result = lhs;
    result /= rhs;
    return result;
}

// Scalar operations
template<typename T, int N>
DualNumber<T, N> operator+(const DualNumber<T, N>& lhs, const T& rhs) {
    return DualNumber<T, N>(lhs.value() + rhs, lhs.derivatives());
}

template<typename T, int N>
DualNumber<T, N> operator+(const T& lhs, const DualNumber<T, N>& rhs) {
    return DualNumber<T, N>(lhs + rhs.value(), rhs.derivatives());
}

template<typename T, int N>
DualNumber<T, N> operator-(const DualNumber<T, N>& lhs, const T& rhs) {
    return DualNumber<T, N>(lhs.value() - rhs, lhs.derivatives());
}

template<typename T, int N>
DualNumber<T, N> operator-(const T& lhs, const DualNumber<T, N>& rhs) {
    DualNumber<T, N> result(lhs - rhs.value());
    for (int i = 0; i < N; ++i) {
        result.derivative(i) = -rhs.derivative(i);
    }
    return result;
}

template<typename T, int N>
DualNumber<T, N> operator*(const DualNumber<T, N>& lhs, const T& rhs) {
    DualNumber<T, N> result = lhs;
    result *= rhs;
    return result;
}

template<typename T, int N>
DualNumber<T, N> operator*(const T& lhs, const DualNumber<T, N>& rhs) {
    DualNumber<T, N> result = rhs;
    result *= lhs;
    return result;
}

template<typename T, int N>
DualNumber<T, N> operator/(const DualNumber<T, N>& lhs, const T& rhs) {
    DualNumber<T, N> result = lhs;
    result /= rhs;
    return result;
}

template<typename T, int N>
DualNumber<T, N> operator/(const T& lhs, const DualNumber<T, N>& rhs) {
    T inv_val = T(1) / rhs.value();
    T inv_val2 = inv_val * inv_val;
    
    DualNumber<T, N> result(lhs * inv_val);
    for (int i = 0; i < N; ++i) {
        result.derivative(i) = -lhs * rhs.derivative(i) * inv_val2;
    }
    return result;
}

// Comparison operators
template<typename T, int N>
bool operator==(const DualNumber<T, N>& lhs, const DualNumber<T, N>& rhs) {
    return lhs.value() == rhs.value();
}

template<typename T, int N>
bool operator!=(const DualNumber<T, N>& lhs, const DualNumber<T, N>& rhs) {
    return lhs.value() != rhs.value();
}

template<typename T, int N>
bool operator<(const DualNumber<T, N>& lhs, const DualNumber<T, N>& rhs) {
    return lhs.value() < rhs.value();
}

template<typename T, int N>
bool operator<=(const DualNumber<T, N>& lhs, const DualNumber<T, N>& rhs) {
    return lhs.value() <= rhs.value();
}

template<typename T, int N>
bool operator>(const DualNumber<T, N>& lhs, const DualNumber<T, N>& rhs) {
    return lhs.value() > rhs.value();
}

template<typename T, int N>
bool operator>=(const DualNumber<T, N>& lhs, const DualNumber<T, N>& rhs) {
    return lhs.value() >= rhs.value();
}

// Mathematical functions
template<typename T, int N>
DualNumber<T, N> sqrt(const DualNumber<T, N>& x) {
    T sqrt_val = std::sqrt(x.value());
    T inv_2sqrt = T(0.5) / sqrt_val;
    
    DualNumber<T, N> result(sqrt_val);
    for (int i = 0; i < N; ++i) {
        result.derivative(i) = x.derivative(i) * inv_2sqrt;
    }
    return result;
}

template<typename T, int N>
DualNumber<T, N> exp(const DualNumber<T, N>& x) {
    T exp_val = std::exp(x.value());
    
    DualNumber<T, N> result(exp_val);
    for (int i = 0; i < N; ++i) {
        result.derivative(i) = x.derivative(i) * exp_val;
    }
    return result;
}

template<typename T, int N>
DualNumber<T, N> log(const DualNumber<T, N>& x) {
    T inv_val = T(1) / x.value();
    
    DualNumber<T, N> result(std::log(x.value()));
    for (int i = 0; i < N; ++i) {
        result.derivative(i) = x.derivative(i) * inv_val;
    }
    return result;
}

template<typename T, int N>
DualNumber<T, N> sin(const DualNumber<T, N>& x) {
    T sin_val = std::sin(x.value());
    T cos_val = std::cos(x.value());
    
    DualNumber<T, N> result(sin_val);
    for (int i = 0; i < N; ++i) {
        result.derivative(i) = x.derivative(i) * cos_val;
    }
    return result;
}

template<typename T, int N>
DualNumber<T, N> cos(const DualNumber<T, N>& x) {
    T sin_val = std::sin(x.value());
    T cos_val = std::cos(x.value());
    
    DualNumber<T, N> result(cos_val);
    for (int i = 0; i < N; ++i) {
        result.derivative(i) = -x.derivative(i) * sin_val;
    }
    return result;
}

template<typename T, int N>
DualNumber<T, N> tan(const DualNumber<T, N>& x) {
    T tan_val = std::tan(x.value());
    T sec2_val = T(1) + tan_val * tan_val;
    
    DualNumber<T, N> result(tan_val);
    for (int i = 0; i < N; ++i) {
        result.derivative(i) = x.derivative(i) * sec2_val;
    }
    return result;
}

template<typename T, int N>
DualNumber<T, N> pow(const DualNumber<T, N>& x, const T& p) {
    T pow_val = std::pow(x.value(), p);
    T dpow_val = p * std::pow(x.value(), p - T(1));
    
    DualNumber<T, N> result(pow_val);
    for (int i = 0; i < N; ++i) {
        result.derivative(i) = x.derivative(i) * dpow_val;
    }
    return result;
}

template<typename T, int N>
DualNumber<T, N> pow(const DualNumber<T, N>& x, const DualNumber<T, N>& y) {
    // x^y = exp(y * log(x))
    return exp(y * log(x));
}

template<typename T, int N>
DualNumber<T, N> abs(const DualNumber<T, N>& x) {
    if (x.value() >= T(0)) {
        return x;
    } else {
        return -x;
    }
}

template<typename T, int N>
DualNumber<T, N> min(const DualNumber<T, N>& a, const DualNumber<T, N>& b) {
    return (a < b) ? a : b;
}

template<typename T, int N>
DualNumber<T, N> max(const DualNumber<T, N>& a, const DualNumber<T, N>& b) {
    return (a > b) ? a : b;
}

// Output stream
template<typename T, int N>
std::ostream& operator<<(std::ostream& os, const DualNumber<T, N>& x) {
    os << "DualNumber(" << x.value() << "; [";
    for (int i = 0; i < N; ++i) {
        if (i > 0) os << ", ";
        os << x.derivative(i);
    }
    os << "])";
    return os;
}

// Type aliases
using DualReal = DualNumber<Real, 1>;
using DualVector3 = DualNumber<Vector3, 1>;

// Multi-dimensional dual numbers for higher derivatives
template<typename T>
using DualNumber2D = DualNumber<T, 2>;

template<typename T>
using DualNumber3D = DualNumber<T, 3>;

// Helper function to create dual numbers
template<typename T, int N>
DualNumber<T, N> makeDual(const T& value, int index = -1) {
    return DualNumber<T, N>(value, index);
}

} // namespace cfd::ad