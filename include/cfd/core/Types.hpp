#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace cfd {

// Precision configuration
#ifdef CFD_DOUBLE_PRECISION
using Real = double;
#else
using Real = float;
#endif

// Integer types
using Index = std::int32_t;
using LocalIndex = std::int32_t;
using GlobalIndex = std::int64_t;

// Mathematical constants
inline constexpr Real PI = Real(3.14159265358979323846);
inline constexpr Real EPSILON = std::numeric_limits<Real>::epsilon();
inline constexpr Real SMALL = Real(1e-20);
inline constexpr Real LARGE = Real(1e20);

// Vector types
using Vector3 = Eigen::Matrix<Real, 3, 1>;
using Vector2 = Eigen::Matrix<Real, 2, 1>;
using VectorX = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

// Matrix types
using Matrix3 = Eigen::Matrix<Real, 3, 3>;
using MatrixX = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<Real>;

// Aligned allocator for SIMD
template<typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

// Smart pointers
template<typename T>
using UniquePtr = std::unique_ptr<T>;

template<typename T>
using SharedPtr = std::shared_ptr<T>;

template<typename T>
using WeakPtr = std::weak_ptr<T>;

// Field types
enum class FieldType {
    SCALAR,
    VECTOR,
    TENSOR
};

// Boundary condition types
enum class BCType {
    DIRICHLET,
    NEUMANN,
    ROBIN,
    PERIODIC,
    SYMMETRY,
    WALL,
    INLET,
    OUTLET,
    FARFIELD
};

// Cell types
enum class CellType {
    TETRAHEDRON,
    HEXAHEDRON,
    PRISM,
    PYRAMID,
    TRIANGLE,
    QUADRILATERAL
};

// Face types
enum class FaceType {
    TRIANGLE,
    QUADRILATERAL,
    INTERNAL,
    BOUNDARY
};

// Solver types
enum class SolverType {
    COMPRESSIBLE,
    INCOMPRESSIBLE,
    LOW_MACH
};

// Time integration schemes
enum class TimeScheme {
    EULER_IMPLICIT,
    EULER_EXPLICIT,
    BDF2,
    BDF3,
    SDIRK3,
    SDIRK4,
    RK4
};

// Flux schemes
enum class FluxScheme {
    UPWIND,
    CENTRAL,
    ROE,
    HLLC,
    AUSM,
    MUSCL,
    WENO5
};

// Gradient schemes
enum class GradientScheme {
    GREEN_GAUSS,
    LEAST_SQUARES,
    WEIGHTED_LEAST_SQUARES
};

// Limiter types
enum class LimiterType {
    NONE,
    BARTH_JESPERSEN,
    VENKATAKRISHNAN,
    MINMOD,
    VANLEER,
    SUPERBEE
};

// Linear solver types
enum class LinearSolverType {
    GMRES,
    BICGSTAB,
    CG,
    DIRECT
};

// Preconditioner types
enum class PreconditionerType {
    NONE,
    JACOBI,
    ILU0,
    ILUK,
    AMG,
    GAUSS_SEIDEL
};

// Turbulence model types
enum class TurbulenceModelType {
    NONE,
    SPALART_ALLMARAS,
    K_EPSILON,
    K_OMEGA,
    K_OMEGA_SST,
    LES_SMAGORINSKY,
    LES_WALE,
    DNS
};

// Utility functions
template<typename T>
inline T sqr(T x) { return x * x; }

template<typename T>
inline T cube(T x) { return x * x * x; }

template<typename T>
inline T sign(T x) { return (x > T(0)) - (x < T(0)); }

template<typename T>
inline T minmod(T a, T b) {
    return sign(a) * std::max(T(0), std::min(std::abs(a), sign(a) * b));
}

// SIMD alignment
constexpr std::size_t SIMD_ALIGNMENT = 32;

// Cache line size
constexpr std::size_t CACHE_LINE_SIZE = 64;

} // namespace cfd