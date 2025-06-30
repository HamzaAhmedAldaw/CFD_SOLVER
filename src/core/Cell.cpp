// ===== CORE MODULE IMPLEMENTATIONS =====

// === src/core/Cell.cpp ===
#include "cfd/core/Cell.hpp"
#include "cfd/core/Face.hpp"
#include <algorithm>
#include <cmath>

namespace cfd {

Cell::Cell(Index id, CellType type)
    : id_(id), type_(type), processor_(0), isGhost_(false) {
}

void Cell::computeGeometry(const std::vector<Vector3>& vertices) {
    // Compute cell center
    center_ = Vector3::Zero();
    for (Index vertexId : vertices_) {
        center_ += vertices[vertexId];
    }
    center_ /= vertices_.size();
    
    // Compute volume based on cell type
    switch (type_) {
        case CellType::TETRAHEDRON:
            computeTetrahedronVolume(vertices);
            break;
        case CellType::HEXAHEDRON:
            computeHexahedronVolume(vertices);
            break;
        case CellType::PRISM:
            computePrismVolume(vertices);
            break;
        case CellType::PYRAMID:
            computePyramidVolume(vertices);
            break;
        default:
            throw std::runtime_error("Unsupported cell type");
    }
}

void Cell::computeTetrahedronVolume(const std::vector<Vector3>& vertices) {
    // Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
    const Vector3& v0 = vertices[vertices_[0]];
    const Vector3& v1 = vertices[vertices_[1]];
    const Vector3& v2 = vertices[vertices_[2]];
    const Vector3& v3 = vertices[vertices_[3]];
    
    Vector3 a = v1 - v0;
    Vector3 b = v2 - v0;
    Vector3 c = v3 - v0;
    
    volume_ = std::abs(a.dot(b.cross(c))) / 6.0;
}

void Cell::computeHexahedronVolume(const std::vector<Vector3>& vertices) {
    // Decompose hexahedron into 6 tetrahedra
    // This is more robust than direct formula for distorted cells
    volume_ = 0.0;
    
    // Define tetrahedra decomposition
    static const int tetras[6][4] = {
        {0, 1, 3, 4}, {1, 2, 3, 6},
        {1, 4, 5, 6}, {3, 4, 6, 7},
        {1, 3, 4, 6}, {4, 5, 6, 7}
    };
    
    for (int t = 0; t < 6; ++t) {
        const Vector3& v0 = vertices[vertices_[tetras[t][0]]];
        const Vector3& v1 = vertices[vertices_[tetras[t][1]]];
        const Vector3& v2 = vertices[vertices_[tetras[t][2]]];
        const Vector3& v3 = vertices[vertices_[tetras[t][3]]];
        
        Vector3 a = v1 - v0;
        Vector3 b = v2 - v0;
        Vector3 c = v3 - v0;
        
        volume_ += std::abs(a.dot(b.cross(c))) / 6.0;
    }
}

void Cell::computePrismVolume(const std::vector<Vector3>& vertices) {
    // Decompose prism into 3 tetrahedra
    volume_ = 0.0;
    
    static const int tetras[3][4] = {
        {0, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}
    };
    
    for (int t = 0; t < 3; ++t) {
        const Vector3& v0 = vertices[vertices_[tetras[t][0]]];
        const Vector3& v1 = vertices[vertices_[tetras[t][1]]];
        const Vector3& v2 = vertices[vertices_[tetras[t][2]]];
        const Vector3& v3 = vertices[vertices_[tetras[t][3]]];
        
        Vector3 a = v1 - v0;
        Vector3 b = v2 - v0;
        Vector3 c = v3 - v0;
        
        volume_ += std::abs(a.dot(b.cross(c))) / 6.0;
    }
}

void Cell::computePyramidVolume(const std::vector<Vector3>& vertices) {
    // Volume = |base_area * height| / 3
    // For general pyramid, decompose into tetrahedra
    volume_ = 0.0;
    
    static const int tetras[2][4] = {
        {0, 1, 2, 4}, {0, 2, 3, 4}
    };
    
    for (int t = 0; t < 2; ++t) {
        const Vector3& v0 = vertices[vertices_[tetras[t][0]]];
        const Vector3& v1 = vertices[vertices_[tetras[t][1]]];
        const Vector3& v2 = vertices[vertices_[tetras[t][2]]];
        const Vector3& v3 = vertices[vertices_[tetras[t][3]]];
        
        Vector3 a = v1 - v0;
        Vector3 b = v2 - v0;
        Vector3 c = v3 - v0;
        
        volume_ += std::abs(a.dot(b.cross(c))) / 6.0;
    }
}

Real Cell::skewness() const {
    // Compute cell skewness (0 = perfect, 1 = highly skewed)
    Real maxSkewness = 0.0;
    
    for (const auto& face : faces_) {
        // Vector from cell center to face center
        Vector3 d = face->center() - center_;
        
        // Face normal
        const Vector3& n = face->normal();
        
        // Angle between d and n
        Real cosTheta = d.normalized().dot(n);
        Real skew = 1.0 - std::abs(cosTheta);
        
        maxSkewness = std::max(maxSkewness, skew);
    }
    
    return maxSkewness;
}

Real Cell::aspectRatio() const {
    // Compute aspect ratio as max edge length / min edge length
    Real minLength = std::numeric_limits<Real>::max();
    Real maxLength = 0.0;
    
    // Get unique edges
    std::set<std::pair<Index, Index>> edges;
    
    for (const auto& face : faces_) {
        const auto& faceVerts = face->vertices();
        for (size_t i = 0; i < faceVerts.size(); ++i) {
            Index v1 = faceVerts[i];
            Index v2 = faceVerts[(i + 1) % faceVerts.size()];
            
            if (v1 > v2) std::swap(v1, v2);
            edges.insert({v1, v2});
        }
    }
    
    // Compute edge lengths
    for (const auto& [v1, v2] : edges) {
        // Need access to vertex positions
        // This is simplified - actual implementation would access mesh vertices
        Real length = 1.0; // Placeholder
        minLength = std::min(minLength, length);
        maxLength = std::max(maxLength, length);
    }
    
    return maxLength / (minLength + SMALL);
}

Real Cell::orthogonality() const {
    // Compute minimum orthogonality quality
    Real minOrthogonality = 1.0;
    
    for (const auto& face : faces_) {
        if (face->isBoundary()) continue;
        
        // Vector between cell centers
        Index neighbor = (face->owner() == id_) ? face->neighbor() : face->owner();
        // Need access to neighbor cell center
        // Vector3 d = neighborCenter - center_;
        Vector3 d(1, 0, 0); // Placeholder
        
        // Face normal
        const Vector3& n = face->normal();
        
        // Orthogonality = |cos(angle)|
        Real ortho = std::abs(d.normalized().dot(n));
        minOrthogonality = std::min(minOrthogonality, ortho);
    }
    
    return minOrthogonality;
}

bool Cell::contains(const Vector3& point) const {
    // Check if point is inside cell using ray casting
    // Simplified implementation - actual would depend on cell type
    
    // For now, use bounding box test
    Vector3 minBound = center_;
    Vector3 maxBound = center_;
    
    // Expand bounds (this is approximate)
    Real radius = std::pow(volume_, 1.0/3.0);
    minBound.array() -= radius;
    maxBound.array() += radius;
    
    return (point.array() >= minBound.array()).all() &&
           (point.array() <= maxBound.array()).all();
}
