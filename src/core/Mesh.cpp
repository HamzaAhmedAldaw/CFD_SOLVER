// === src/core/Mesh.cpp ===
#include "cfd/core/Mesh.hpp"
#include "cfd/io/Logger.hpp"
#include <algorithm>
#include <unordered_map>
#include <set>

namespace cfd {

Mesh::Mesh() {
    logger_ = std::make_shared<io::Logger>("Mesh");
}

Mesh::~Mesh() = default;

void Mesh::addCell(CellType type, const std::vector<Index>& vertices) {
    Index cellId = cells_.size();
    cells_.emplace_back(std::make_unique<Cell>(cellId, type));
    
    Cell& cell = *cells_.back();
    cell.setVertices(vertices);
}

void Mesh::build(const std::vector<Vector3>& vertices) {
    vertices_ = vertices;
    logger_->info("Building mesh connectivity...");
    
    // Build faces from cells
    buildFaces();
    
    // Compute geometry
    computeGeometry();
    
    // Build additional connectivity
    buildCellConnectivity();
    
    logger_->info("Mesh build complete: {} cells, {} faces, {} vertices",
                  numCells(), numFaces(), numVertices());
}

void Mesh::buildFaces() {
    logger_->debug("Building faces from cells");
    
    // Map to track unique faces
    struct FaceHash {
        size_t operator()(const std::vector<Index>& verts) const {
            // Sort vertices for consistent hashing
            std::vector<Index> sorted = verts;
            std::sort(sorted.begin(), sorted.end());
            
            size_t hash = 0;
            for (Index v : sorted) {
                hash ^= std::hash<Index>()(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
    
    std::unordered_map<std::vector<Index>, Index, FaceHash> faceMap;
    
    // Process each cell
    for (Index cellId = 0; cellId < numCells(); ++cellId) {
        Cell& cell = *cells_[cellId];
        
        // Get cell faces based on type
        std::vector<std::vector<Index>> cellFaces = getCellFaces(cell);
        
        for (const auto& faceVerts : cellFaces) {
            // Sort vertices for comparison
            std::vector<Index> sortedVerts = faceVerts;
            std::sort(sortedVerts.begin(), sortedVerts.end());
            
            auto it = faceMap.find(sortedVerts);
            if (it == faceMap.end()) {
                // New face
                Index faceId = faces_.size();
                faceMap[sortedVerts] = faceId;
                
                FaceType ftype = (faceVerts.size() == 3) ? 
                                FaceType::TRIANGLE : FaceType::QUADRILATERAL;
                faces_.emplace_back(std::make_unique<Face>(faceId, ftype));
                
                Face& face = *faces_.back();
                face.setVertices(faceVerts);
                face.setOwner(cellId);
                
                cell.addFace(faces_.back().get());
            } else {
                // Existing face - set neighbor
                Index faceId = it->second;
                Face& face = *faces_[faceId];
                face.setNeighbor(cellId);
                
                cell.addFace(faces_[faceId].get());
            }
        }
    }
    
    // Mark boundary faces
    for (auto& face : faces_) {
        if (face->neighbor() < 0) {
            face->setType(FaceType::BOUNDARY);
        } else {
            face->setType(FaceType::INTERNAL);
        }
    }
}

std::vector<std::vector<Index>> Mesh::getCellFaces(const Cell& cell) const {
    std::vector<std::vector<Index>> faces;
    const auto& verts = cell.vertices();
    
    switch (cell.type()) {
        case CellType::TETRAHEDRON:
            faces = {{verts[0], verts[1], verts[2]},
                    {verts[0], verts[1], verts[3]},
                    {verts[1], verts[2], verts[3]},
                    {verts[2], verts[0], verts[3]}};
            break;
            
        case CellType::HEXAHEDRON:
            faces = {{verts[0], verts[1], verts[2], verts[3]},
                    {verts[4], verts[5], verts[6], verts[7]},
                    {verts[0], verts[1], verts[5], verts[4]},
                    {verts[2], verts[3], verts[7], verts[6]},
                    {verts[0], verts[3], verts[7], verts[4]},
                    {verts[1], verts[2], verts[6], verts[5]}};
            break;
            
        case CellType::PRISM:
            faces = {{verts[0], verts[1], verts[2]},
                    {verts[3], verts[4], verts[5]},
                    {verts[0], verts[1], verts[4], verts[3]},
                    {verts[1], verts[2], verts[5], verts[4]},
                    {verts[2], verts[0], verts[3], verts[5]}};
            break;
            
        case CellType::PYRAMID:
            faces = {{verts[0], verts[1], verts[2], verts[3]},
                    {verts[0], verts[1], verts[4]},
                    {verts[1], verts[2], verts[4]},
                    {verts[2], verts[3], verts[4]},
                    {verts[3], verts[0], verts[4]}};
            break;
            
        default:
            throw std::runtime_error("Unsupported cell type");
    }
    
    return faces;
}

void Mesh::computeGeometry() {
    logger_->debug("Computing mesh geometry");
    
    // Compute face geometry
    for (auto& face : faces_) {
        face->computeGeometry(vertices_);
    }
    
    // Compute cell geometry
    for (auto& cell : cells_) {
        cell->computeGeometry(vertices_);
    }
}

void Mesh::buildCellConnectivity() {
    logger_->debug("Building cell connectivity");
    
    // Build cell-to-cell connectivity through faces
    for (const auto& face : faces_) {
        if (!face->isBoundary()) {
            Index owner = face->owner();
            Index neighbor = face->neighbor();
            
            cells_[owner]->addNeighbor(neighbor);
            cells_[neighbor]->addNeighbor(owner);
        }
    }
}

void Mesh::addBoundaryPatch(const std::string& name, BCType type) {
    boundaryPatches_.emplace_back(name, type);
}

BoundaryPatch& Mesh::boundaryPatch(const std::string& name) {
    auto it = std::find_if(boundaryPatches_.begin(), boundaryPatches_.end(),
                          [&name](const BoundaryPatch& patch) {
                              return patch.name() == name;
                          });
    
    if (it == boundaryPatches_.end()) {
        throw std::runtime_error("Boundary patch not found: " + name);
    }
    
    return *it;
}

void Mesh::scale(Real factor) {
    logger_->info("Scaling mesh by factor {}", factor);
    
    // Scale vertices
    for (auto& vertex : vertices_) {
        vertex *= factor;
    }
    
    // Recompute geometry
    computeGeometry();
}

void Mesh::renumberCells(const std::vector<Index>& newOrder) {
    logger_->info("Renumbering cells");
    
    if (newOrder.size() != numCells()) {
        throw std::runtime_error("Invalid renumbering array size");
    }
    
    // Create new cell array
    std::vector<std::unique_ptr<Cell>> newCells(numCells());
    
    for (Index oldId = 0; oldId < numCells(); ++oldId) {
        Index newId = newOrder[oldId];
        newCells[newId] = std::move(cells_[oldId]);
        newCells[newId]->setId(newId);
    }
    
    cells_ = std::move(newCells);
    
    // Update face owner/neighbor references
    std::vector<Index> reverseMap(numCells());
    for (Index i = 0; i < numCells(); ++i) {
        reverseMap[newOrder[i]] = i;
    }
    
    for (auto& face : faces_) {
        face->setOwner(newOrder[face->owner()]);
        if (!face->isBoundary()) {
            face->setNeighbor(newOrder[face->neighbor()]);
        }
    }
}

Index Mesh::findCell(const Vector3& point) const {
    // Simple linear search - could be optimized with spatial data structure
    for (Index i = 0; i < numCells(); ++i) {
        if (cells_[i]->contains(point)) {
            return i;
        }
    }
    
    return -1; // Not found
}

Real Mesh::quality() const {
    Real minQuality = 1.0;
    
    for (const auto& cell : cells_) {
        Real skewness = cell->skewness();
        Real aspectRatio = cell->aspectRatio();
        Real orthogonality = cell->orthogonality();
        
        // Combined quality metric
        Real quality = orthogonality * (1.0 - skewness) / aspectRatio;
        minQuality = std::min(minQuality, quality);
    }
    
    return minQuality;
}
