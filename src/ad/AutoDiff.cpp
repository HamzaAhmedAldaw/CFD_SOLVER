// ===== AUTOMATIC DIFFERENTIATION IMPLEMENTATIONS =====

// === src/ad/AutoDiff.cpp ===
#include "cfd/ad/AutoDiff.hpp"

namespace cfd::ad {

// Static member initialization
const std::vector<Index> CommunicationPattern::emptyVector_;

// Explicit template instantiations for common types
template class FieldAD<Real>;
template class FieldAD<Vector3>;

// Implementation of utility functions for AD verification
namespace {
    // Helper function for numerical gradient checking
    bool compareGradients(const VectorX& grad1, const VectorX& grad2, Real tol) {
        if (grad1.size() != grad2.size()) return false;
        
        Real maxError = 0.0;
        for (Index i = 0; i < grad1.size(); ++i) {
            Real error = std::abs(grad1[i] - grad2[i]);
            Real scale = std::max(std::abs(grad1[i]), std::abs(grad2[i]));
            if (scale > EPSILON) {
                error /= scale;
            }
            maxError = std::max(maxError, error);
        }
        
        return maxError < tol;
    }
}

// Explicit instantiation of commonly used templates
template VectorX ForwardAD::gradient(std::function<Real(const std::vector<DualReal>&)>, const VectorX&);
template MatrixX ForwardAD::jacobian(std::function<std::vector<DualReal>(const std::vector<DualReal>&)>, const VectorX&);

} // namespace cfd::ad
