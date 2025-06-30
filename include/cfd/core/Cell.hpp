#pragma once

#include "cfd/core/Types.hpp"
#include <vector>
#include <array>

namespace cfd {

class Cell {
public:
    // Constructors
    Cell() = default;
    Cell(CellType type, const std::vector<Index>& nodeIds);
    
    // Cell properties
    CellType type() const { return type_; }
    Index id() const { return id_; }
    void setId(Index id) { id_ = id; }
    
    // Geometric properties
    const Vector3& center() const { return center_; }
    Real volume() const { return volume_; }
    const Matrix3& inertia() const { return inertia_; }
    
    // Topology
    const std::vector<Index>& nodes() const { return nodeIds_; }
    const std::vector<Index>& faces() const { return faceIds_; }
    const std::vector<Index>& neighbors() const { return neighborIds_; }
    
    Index numNodes() const { return nodeIds_.size(); }
    Index numFaces() const { return faceIds_.size(); }
    Index numNeighbors() const { return neighborIds_.size(); }
    
    // Connectivity
    void addFace(Index faceId) { faceIds_.push_back(faceId); }
    void addNeighbor(Index neighborId) { neighborIds_.push_back(neighborId); }
    
    // Geometry computation
    void computeGeometry(const std::vector<Vector3>& meshNodes);
    void computeCenter(const std::vector<Vector3>& meshNodes);
    void computeVolume(const std::vector<Vector3>& meshNodes);
    void computeInertia(const std::vector<Vector3>& meshNodes);
    
    // Quality metrics
    Real aspectRatio() const;
    Real skewness() const;
    Real orthogonality() const;
    
    // Parallel info
    Index processor() const { return processor_; }
    void setProcessor(Index proc) { processor_ = proc; }
    bool isGhost() const { return isGhost_; }
    void setGhost(bool ghost) { isGhost_ = ghost; }
    
    // Interpolation weights for gradients
    struct GradientWeights {
        AlignedVector<Real> lsqWeights;    // Least squares weights
        AlignedVector<Vector3> lsqVectors; // Distance vectors
        Matrix3 lsqMatrix;                 // Pre-computed matrix
        bool valid = false;
    };
    
    const GradientWeights& gradientWeights() const { return gradWeights_; }
    void computeGradientWeights(const std::vector<Cell>& cells);

private:
    // Basic properties
    CellType type_ = CellType::TETRAHEDRON;
    Index id_ = -1;
    
    // Geometric properties
    Vector3 center_ = Vector3::Zero();
    Real volume_ = 0.0;
    Matrix3 inertia_ = Matrix3::Zero();
    
    // Topology
    std::vector<Index> nodeIds_;
    std::vector<Index> faceIds_;
    std::vector<Index> neighborIds_;
    
    // Parallel
    Index processor_ = 0;
    bool isGhost_ = false;
    
    // Cached data for efficiency
    mutable GradientWeights gradWeights_;
    
    // Helper functions for volume calculation
    Real computeTetVolume(const std::array<Vector3, 4>& nodes) const;
    Real computeHexVolume(const std::array<Vector3, 8>& nodes) const;
    Real computePrismVolume(const std::array<Vector3, 6>& nodes) const;
    Real computePyramidVolume(const std::array<Vector3, 5>& nodes) const;
};

// Inline implementations for performance
inline Real Cell::aspectRatio() const {
    // Simplified calculation - ratio of max to min eigenvalue of inertia
    Eigen::SelfAdjointEigenSolver<Matrix3> solver(inertia_);
    Vector3 eigenvalues = solver.eigenvalues();
    return eigenvalues.maxCoeff() / (eigenvalues.minCoeff() + SMALL);
}

} // namespace cfd