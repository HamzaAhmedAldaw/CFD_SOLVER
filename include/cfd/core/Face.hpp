#pragma once

#include "cfd/core/Types.hpp"
#include <vector>
#include <string>

namespace cfd {

class Face {
public:
    // Constructors
    Face() = default;
    Face(FaceType type, const std::vector<Index>& nodeIds);
    
    // Face properties
    FaceType type() const { return type_; }
    Index id() const { return id_; }
    void setId(Index id) { id_ = id; }
    
    // Geometric properties
    const Vector3& center() const { return center_; }
    const Vector3& normal() const { return normal_; }
    Real area() const { return area_; }
    
    // Topology
    const std::vector<Index>& nodes() const { return nodeIds_; }
    Index numNodes() const { return nodeIds_.size(); }
    
    // Connectivity
    Index owner() const { return ownerCell_; }
    Index neighbour() const { return neighbourCell_; }
    void setOwner(Index cellId) { ownerCell_ = cellId; }
    void setNeighbour(Index cellId) { neighbourCell_ = cellId; }
    
    // Boundary information
    bool isInternal() const { return neighbourCell_ >= 0; }
    bool isBoundary() const { return neighbourCell_ < 0; }
    const std::string& patch() const { return patchName_; }
    void setPatch(const std::string& name) { patchName_ = name; }
    
    // Geometry computation
    void computeGeometry(const std::vector<Vector3>& meshNodes);
    void computeNormal(const std::vector<Vector3>& meshNodes);
    void computeArea(const std::vector<Vector3>& meshNodes);
    void computeCenter(const std::vector<Vector3>& meshNodes);
    
    // Interpolation factors
    Real weight() const { return interpolationWeight_; }
    Real deltaCoeff() const { return deltaCoeff_; }
    void computeInterpolationFactors(const Vector3& ownerCenter, 
                                    const Vector3& neighbourCenter);
    
    // Skewness and orthogonality
    Real skewness() const { return skewness_; }
    Real nonOrthogonality() const { return nonOrthogonality_; }
    void computeQualityMetrics(const Vector3& ownerCenter,
                              const Vector3& neighbourCenter);
    
    // For non-orthogonal correction
    const Vector3& delta() const { return delta_; }
    const Vector3& correctionVector() const { return correctionVector_; }
    
    // Parallel communication
    bool isProcessor() const { return isProcessor_; }
    void setProcessor(bool proc) { isProcessor_ = proc; }
    Index processorPatch() const { return processorPatch_; }
    void setProcessorPatch(Index patch) { processorPatch_ = patch; }

private:
    // Basic properties
    FaceType type_ = FaceType::TRIANGLE;
    Index id_ = -1;
    
    // Geometric properties
    Vector3 center_ = Vector3::Zero();
    Vector3 normal_ = Vector3::Zero();  // Unit normal pointing from owner to neighbour
    Real area_ = 0.0;
    
    // Topology
    std::vector<Index> nodeIds_;
    Index ownerCell_ = -1;
    Index neighbourCell_ = -1;
    std::string patchName_;
    
    // Interpolation data
    Real interpolationWeight_ = 0.5;  // Linear interpolation weight
    Real deltaCoeff_ = 0.0;          // 1/|d| for gradient computation
    Vector3 delta_ = Vector3::Zero(); // Vector from owner to neighbour centers
    
    // Quality metrics
    Real skewness_ = 0.0;
    Real nonOrthogonality_ = 0.0;
    Vector3 correctionVector_ = Vector3::Zero();
    
    // Parallel
    bool isProcessor_ = false;
    Index processorPatch_ = -1;
    
    // Helper functions
    Real computeTriangleArea(const Vector3& v0, const Vector3& v1, const Vector3& v2) const;
    Real computeQuadArea(const Vector3& v0, const Vector3& v1, 
                        const Vector3& v2, const Vector3& v3) const;
};

// Inline implementations
inline void Face::computeInterpolationFactors(const Vector3& ownerCenter,
                                             const Vector3& neighbourCenter) {
    delta_ = neighbourCenter - ownerCenter;
    Real magDelta = delta_.norm();
    deltaCoeff_ = 1.0 / (magDelta + SMALL);
    
    // Linear interpolation weight based on distance to face
    Real distOwner = (center_ - ownerCenter).norm();
    interpolationWeight_ = distOwner / (magDelta + SMALL);
}

inline void Face::computeQualityMetrics(const Vector3& ownerCenter,
                                       const Vector3& neighbourCenter) {
    Vector3 d = neighbourCenter - ownerCenter;
    Vector3 dUnit = d.normalized();
    
    // Non-orthogonality angle
    Real cosTheta = std::abs(dUnit.dot(normal_));
    nonOrthogonality_ = std::acos(std::min(cosTheta, Real(1.0))) * 180.0 / PI;
    
    // Skewness - deviation of face center from the line joining cell centers
    Vector3 fi = center_ - ownerCenter;
    Real fid = fi.dot(dUnit);
    Vector3 faceOnLine = ownerCenter + fid * dUnit;
    skewness_ = (center_ - faceOnLine).norm() / d.norm();
    
    // Correction vector for non-orthogonal meshes
    correctionVector_ = normal_ - dUnit * (dUnit.dot(normal_));
}

} // namespace cfd