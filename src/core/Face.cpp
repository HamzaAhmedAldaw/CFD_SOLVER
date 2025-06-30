// === src/core/Face.cpp ===
#include "cfd/core/Face.hpp"
#include <cmath>

namespace cfd {

Face::Face(Index id, FaceType type)
    : id_(id), type_(type), owner_(-1), neighbor_(-1),
      patchId_(-1), area_(0.0) {
}

void Face::computeGeometry(const std::vector<Vector3>& vertices) {
    // Compute face center
    center_ = Vector3::Zero();
    for (Index vertexId : vertices_) {
        center_ += vertices[vertexId];
    }
    center_ /= vertices_.size();
    
    // Compute area and normal
    if (vertices_.size() == 3) {
        // Triangle
        computeTriangleGeometry(vertices);
    } else if (vertices_.size() == 4) {
        // Quadrilateral
        computeQuadGeometry(vertices);
    } else {
        // General polygon
        computePolygonGeometry(vertices);
    }
    
    // Ensure unit normal
    normal_.normalize();
}

void Face::computeTriangleGeometry(const std::vector<Vector3>& vertices) {
    const Vector3& v0 = vertices[vertices_[0]];
    const Vector3& v1 = vertices[vertices_[1]];
    const Vector3& v2 = vertices[vertices_[2]];
    
    Vector3 a = v1 - v0;
    Vector3 b = v2 - v0;
    
    Vector3 cross = a.cross(b);
    area_ = 0.5 * cross.norm();
    normal_ = cross.normalized();
}

void Face::computeQuadGeometry(const std::vector<Vector3>& vertices) {
    // Decompose into two triangles
    const Vector3& v0 = vertices[vertices_[0]];
    const Vector3& v1 = vertices[vertices_[1]];
    const Vector3& v2 = vertices[vertices_[2]];
    const Vector3& v3 = vertices[vertices_[3]];
    
    // First triangle (0,1,2)
    Vector3 a1 = v1 - v0;
    Vector3 b1 = v2 - v0;
    Vector3 cross1 = a1.cross(b1);
    
    // Second triangle (0,2,3)
    Vector3 a2 = v2 - v0;
    Vector3 b2 = v3 - v0;
    Vector3 cross2 = a2.cross(b2);
    
    // Total area and average normal
    area_ = 0.5 * (cross1.norm() + cross2.norm());
    normal_ = (cross1 + cross2).normalized();
}

void Face::computePolygonGeometry(const std::vector<Vector3>& vertices) {
    // Use fan triangulation from centroid
    area_ = 0.0;
    normal_ = Vector3::Zero();
    
    for (size_t i = 0; i < vertices_.size(); ++i) {
        const Vector3& v0 = center_;
        const Vector3& v1 = vertices[vertices_[i]];
        const Vector3& v2 = vertices[vertices_[(i + 1) % vertices_.size()]];
        
        Vector3 a = v1 - v0;
        Vector3 b = v2 - v0;
        Vector3 cross = a.cross(b);
        
        area_ += 0.5 * cross.norm();
        normal_ += cross;
    }
    
    normal_.normalize();
}

Real Face::interpolationWeight() const {
    // Linear interpolation weight for owner/neighbor
    // Weight = distance to neighbor / total distance
    
    // This requires access to cell centers
    // Simplified implementation returns 0.5
    return 0.5;
}

Real Face::nonOrthogonality() const {
    // Angle between face normal and line connecting cell centers
    if (isBoundary()) return 0.0;
    
    // Need access to cell centers
    // Vector3 d = neighborCenter - ownerCenter;
    Vector3 d(1, 0, 0); // Placeholder
    
    Real cosTheta = std::abs(d.normalized().dot(normal_));
    return std::acos(cosTheta) * 180.0 / PI; // Return in degrees
}
