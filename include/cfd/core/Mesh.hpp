#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Cell.hpp"
#include "cfd/core/Face.hpp"
#include "cfd/memory/MemoryPool.hpp"
#include <unordered_map>
#include <vector>
#include <string>

namespace cfd {

class Mesh {
public:
    // Constructor/Destructor
    Mesh();
    ~Mesh();
    
    // Mesh statistics
    struct Stats {
        Index numCells;
        Index numFaces;
        Index numNodes;
        Index numBoundaryFaces;
        Index numInternalFaces;
        Real minVolume;
        Real maxVolume;
        Real minQuality;
        Real maxAspectRatio;
        Real totalVolume;
    };
    
    // Initialize mesh from file
    void readFromFile(const std::string& filename);
    
    // Mesh generation
    void generateCartesian(const Vector3& min, const Vector3& max,
                          const Eigen::Vector3i& divisions);
    
    // Access functions
    Index numCells() const { return cells_.size(); }
    Index numFaces() const { return faces_.size(); }
    Index numNodes() const { return nodes_.size(); }
    Index numInternalFaces() const { return numInternalFaces_; }
    Index numBoundaryFaces() const { return numBoundaryFaces_; }
    
    // Element access
    Cell& cell(Index i) { return cells_[i]; }
    const Cell& cell(Index i) const { return cells_[i]; }
    
    Face& face(Index i) { return faces_[i]; }
    const Face& face(Index i) const { return faces_[i]; }
    
    Vector3& node(Index i) { return nodes_[i]; }
    const Vector3& node(Index i) const { return nodes_[i]; }
    
    // Boundary access
    const std::vector<Index>& boundaryFaces(const std::string& name) const {
        return boundaries_.at(name);
    }
    
    std::vector<std::string> boundaryNames() const;
    
    // Mesh metrics computation
    void computeMetrics();
    Stats getStats() const;
    
    // Mesh quality checks
    bool checkQuality(Real minQuality = 0.1) const;
    void reportQuality() const;
    
    // Mesh refinement
    void refineUniform(int levels = 1);
    void refineAdaptive(const std::vector<Real>& errorEstimate,
                       Real refinementThreshold);
    
    // Parallel domain decomposition
    void decompose(int numDomains);
    
    // Ghost cell management
    void createGhostCells();
    void updateGhostCells();
    
    // Memory pool access
    memory::MemoryPool& memoryPool() { return memoryPool_; }
    
private:
    // Mesh data
    AlignedVector<Cell> cells_;
    AlignedVector<Face> faces_;
    AlignedVector<Vector3> nodes_;
    
    // Boundary information
    std::unordered_map<std::string, std::vector<Index>> boundaries_;
    
    // Mesh topology
    std::vector<std::vector<Index>> cellToCell_;
    std::vector<std::vector<Index>> cellToFace_;
    std::vector<std::vector<Index>> nodeToCell_;
    
    // Internal/boundary face split
    Index numInternalFaces_;
    Index numBoundaryFaces_;
    
    // Domain decomposition
    std::vector<Index> cellDomain_;
    std::vector<Index> ghostCells_;
    
    // Memory management
    memory::MemoryPool memoryPool_;
    
    // Helper functions
    void buildConnectivity();
    void reorderFaces();
    void computeCellVolumes();
    void computeFaceAreas();
    void computeCellCenters();
    Real computeCellQuality(Index cellId) const;
    
    // File I/O helpers
    void readOpenFOAM(const std::string& caseDir);
    void readCGNS(const std::string& filename);
    void readVTK(const std::string& filename);
};

// Inline implementations
inline std::vector<std::string> Mesh::boundaryNames() const {
    std::vector<std::string> names;
    names.reserve(boundaries_.size());
    for (const auto& [name, _] : boundaries_) {
        names.push_back(name);
    }
    return names;
}

} // namespace cfd