// === src/parallel/Domain.cpp ===
#include "cfd/parallel/Domain.hpp"
#include "cfd/parallel/MPI_Wrapper.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cfd::parallel {

void DomainDecomposition::decompose(Mesh& mesh) {
    if (!MPIWrapper::isParallel()) {
        // Single processor - all cells belong to rank 0
        for (Index i = 0; i < mesh.numCells(); ++i) {
            mesh.cell(i).setProcessor(0);
        }
        return;
    }
    
    const int numProcs = MPIWrapper::size();
    
    switch (method_) {
        case Method::SIMPLE:
            simpleDecomposition(mesh, numProcs);
            break;
        case Method::RCB:
            recursiveCoordinateBisection(mesh, numProcs);
            break;
        case Method::GRAPH:
            graphPartitioning(mesh, numProcs);
            break;
        case Method::METIS:
#ifdef CFD_USE_METIS
            metisPartitioning(mesh, numProcs);
#else
            // Fallback to RCB if METIS not available
            recursiveCoordinateBisection(mesh, numProcs);
#endif
            break;
    }
    
    // Update ghost cells and interfaces
    updateGhostCells(mesh);
    computeLoadBalance(mesh);
}

void DomainDecomposition::simpleDecomposition(Mesh& mesh, int numProcs) {
    const Index numCells = mesh.numCells();
    const Index cellsPerProc = numCells / numProcs;
    const Index remainder = numCells % numProcs;
    
    Index cellId = 0;
    for (int proc = 0; proc < numProcs; ++proc) {
        Index numCellsThisProc = cellsPerProc + (proc < remainder ? 1 : 0);
        
        for (Index i = 0; i < numCellsThisProc; ++i) {
            mesh.cell(cellId++).setProcessor(proc);
        }
    }
}

void DomainDecomposition::recursiveCoordinateBisection(Mesh& mesh, int numProcs) {
    // Compute cell centers
    std::vector<Vector3> centers(mesh.numCells());
    for (Index i = 0; i < mesh.numCells(); ++i) {
        centers[i] = mesh.cell(i).center();
    }
    
    // Create initial partition with all cells
    std::vector<std::vector<Index>> partitions(1);
    partitions[0].resize(mesh.numCells());
    std::iota(partitions[0].begin(), partitions[0].end(), 0);
    
    // Recursive bisection
    while (partitions.size() < static_cast<size_t>(numProcs)) {
        std::vector<std::vector<Index>> newPartitions;
        
        for (auto& partition : partitions) {
            if (partition.size() <= 1) {
                newPartitions.push_back(std::move(partition));
                continue;
            }
            
            // Find cutting direction (largest extent)
            Vector3 minCoord = centers[partition[0]];
            Vector3 maxCoord = centers[partition[0]];
            
            for (Index cellId : partition) {
                const Vector3& center = centers[cellId];
                for (int d = 0; d < 3; ++d) {
                    minCoord[d] = std::min(minCoord[d], center[d]);
                    maxCoord[d] = std::max(maxCoord[d], center[d]);
                }
            }
            
            Vector3 extent = maxCoord - minCoord;
            int cutDir = 0;
            Real maxExtent = extent[0];
            for (int d = 1; d < 3; ++d) {
                if (extent[d] > maxExtent) {
                    maxExtent = extent[d];
                    cutDir = d;
                }
            }
            
            // Find median cut position
            std::vector<Real> coords;
            coords.reserve(partition.size());
            for (Index cellId : partition) {
                coords.push_back(centers[cellId][cutDir]);
            }
            
            std::nth_element(coords.begin(), 
                           coords.begin() + coords.size()/2,
                           coords.end());
            Real cutPos = coords[coords.size()/2];
            
            // Split partition
            std::vector<Index> left, right;
            for (Index cellId : partition) {
                if (centers[cellId][cutDir] < cutPos) {
                    left.push_back(cellId);
                } else {
                    right.push_back(cellId);
                }
            }
            
            // Handle edge case where all cells end up on one side
            if (left.empty() || right.empty()) {
                size_t half = partition.size() / 2;
                left.assign(partition.begin(), partition.begin() + half);
                right.assign(partition.begin() + half, partition.end());
            }
            
            newPartitions.push_back(std::move(left));
            newPartitions.push_back(std::move(right));
        }
        
        partitions = std::move(newPartitions);
    }
    
    // Assign processors to cells
    for (int proc = 0; proc < numProcs; ++proc) {
        for (Index cellId : partitions[proc]) {
            mesh.cell(cellId).setProcessor(proc);
        }
    }
}

void DomainDecomposition::graphPartitioning(Mesh& mesh, int numProcs) {
    // Build dual graph of mesh (cell connectivity)
    std::vector<std::vector<Index>> graph(mesh.numCells());
    
    for (Index faceId = 0; faceId < mesh.numFaces(); ++faceId) {
        const Face& face = mesh.face(faceId);
        if (!face.isBoundary()) {
            graph[face.owner()].push_back(face.neighbor());
            graph[face.neighbor()].push_back(face.owner());
        }
    }
    
    // Simple graph partitioning using breadth-first search
    std::vector<int> partition(mesh.numCells(), -1);
    std::vector<Index> partitionSizes(numProcs, 0);
    
    const Index targetSize = mesh.numCells() / numProcs;
    
    // Start from multiple seed points
    std::vector<Index> seeds(numProcs);
    for (int i = 0; i < numProcs; ++i) {
        seeds[i] = (mesh.numCells() * i) / numProcs;
    }
    
    // Grow partitions from seeds
    std::vector<std::queue<Index>> queues(numProcs);
    for (int proc = 0; proc < numProcs; ++proc) {
        partition[seeds[proc]] = proc;
        partitionSizes[proc] = 1;
        queues[proc].push(seeds[proc]);
    }
    
    // BFS growth
    bool growing = true;
    while (growing) {
        growing = false;
        
        for (int proc = 0; proc < numProcs; ++proc) {
            if (partitionSizes[proc] >= targetSize) continue;
            
            std::queue<Index>& queue = queues[proc];
            size_t levelSize = queue.size();
            
            for (size_t i = 0; i < levelSize; ++i) {
                if (partitionSizes[proc] >= targetSize) break;
                
                Index cellId = queue.front();
                queue.pop();
                
                for (Index neighbor : graph[cellId]) {
                    if (partition[neighbor] == -1) {
                        partition[neighbor] = proc;
                        partitionSizes[proc]++;
                        queue.push(neighbor);
                        growing = true;
                    }
                }
            }
        }
    }
    
    // Assign remaining cells
    int currentProc = 0;
    for (Index i = 0; i < mesh.numCells(); ++i) {
        if (partition[i] == -1) {
            partition[i] = currentProc;
            currentProc = (currentProc + 1) % numProcs;
        }
        mesh.cell(i).setProcessor(partition[i]);
    }
}

#ifdef CFD_USE_METIS
void DomainDecomposition::metisPartitioning(Mesh& mesh, int numProcs) {
    // Implementation would use METIS library
    // This is a placeholder - actual implementation would call METIS_PartGraphKway
    recursiveCoordinateBisection(mesh, numProcs);
}
#endif

void DomainDecomposition::updateGhostCells(Mesh& mesh) {
    const int myRank = MPIWrapper::rank();
    
    // Mark cells as ghost if they neighbor cells from other processors
    for (Index cellId = 0; cellId < mesh.numCells(); ++cellId) {
        Cell& cell = mesh.cell(cellId);
        
        if (cell.processor() != myRank) {
            cell.setGhost(true);
            continue;
        }
        
        // Check if any neighbor is from different processor
        bool hasRemoteNeighbor = false;
        for (const auto& face : cell.faces()) {
            if (!face.isBoundary()) {
                Index neighborId = (face.owner() == cellId) ? 
                                 face.neighbor() : face.owner();
                if (mesh.cell(neighborId).processor() != myRank) {
                    hasRemoteNeighbor = true;
                    break;
                }
            }
        }
        
        cell.setGhost(hasRemoteNeighbor);
    }
}

void DomainDecomposition::computeLoadBalance(const Mesh& mesh) {
    const int myRank = MPIWrapper::rank();
    
    // Count local cells
    Index localCells = 0;
    Index localFaces = 0;
    
    for (Index i = 0; i < mesh.numCells(); ++i) {
        if (mesh.cell(i).processor() == myRank && !mesh.cell(i).isGhost()) {
            localCells++;
        }
    }
    
    for (Index i = 0; i < mesh.numFaces(); ++i) {
        const Face& face = mesh.face(i);
        if (mesh.cell(face.owner()).processor() == myRank) {
            localFaces++;
        }
    }
    
    // Gather load information
    loadInfo_.rank = myRank;
    loadInfo_.numCells = localCells;
    loadInfo_.numFaces = localFaces;
    loadInfo_.load = static_cast<Real>(localCells);  // Simple load metric
    
    // Compute global statistics
    if (MPIWrapper::isParallel()) {
        Index globalCells = globalSum(localCells);
        Real avgCells = static_cast<Real>(globalCells) / MPIWrapper::size();
        Real maxCells = globalMax(static_cast<Real>(localCells));
        
        loadImbalance_ = (maxCells / avgCells) - 1.0;
    } else {
        loadImbalance_ = 0.0;
    }
}

Real DomainDecomposition::getCutQuality(const Mesh& mesh) const {
    Index cutEdges = 0;
    Index totalEdges = 0;
    
    for (Index faceId = 0; faceId < mesh.numFaces(); ++faceId) {
        const Face& face = mesh.face(faceId);
        if (!face.isBoundary()) {
            totalEdges++;
            
            int proc1 = mesh.cell(face.owner()).processor();
            int proc2 = mesh.cell(face.neighbor()).processor();
            
            if (proc1 != proc2) {
                cutEdges++;
            }
        }
    }
    
    Index globalCutEdges = globalSum(cutEdges);
    Index globalTotalEdges = globalSum(totalEdges);
    
    return (globalTotalEdges > 0) ? 
           static_cast<Real>(globalCutEdges) / globalTotalEdges : 0.0;
}

} // namespace cfd::parallel
