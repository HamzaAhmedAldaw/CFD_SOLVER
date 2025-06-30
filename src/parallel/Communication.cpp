// ===== PARALLEL IMPLEMENTATIONS =====

// === src/parallel/Communication.cpp ===
#include "cfd/parallel/Communication.hpp"
#include "cfd/parallel/MPI_Wrapper.hpp"
#include <algorithm>
#include <unordered_set>
#include <numeric>

namespace cfd::parallel {

void CommunicationPattern::build() {
    if (!MPIWrapper::isParallel()) return;
    
    findProcessorInterfaces();
    buildGhostCellLists();
}

void CommunicationPattern::findProcessorInterfaces() {
    const int myRank = MPIWrapper::rank();
    const int numProcs = MPIWrapper::size();
    
    // Map to store interfaces by neighbor rank
    std::map<int, ProcessorInterface> interfaceMap;
    
    // First pass: identify neighboring processors
    std::unordered_set<int> neighborRanks;
    
    for (Index faceId = 0; faceId < mesh_.numFaces(); ++faceId) {
        const Face& face = mesh_.face(faceId);
        
        if (face.isBoundary()) continue;
        
        const Cell& owner = mesh_.cell(face.owner());
        const Cell& neighbor = mesh_.cell(face.neighbor());
        
        // Check if face connects to different processor
        if (owner.processor() != neighbor.processor()) {
            int otherRank = (owner.processor() == myRank) ? 
                           neighbor.processor() : owner.processor();
            neighborRanks.insert(otherRank);
            
            // Initialize interface if needed
            if (interfaceMap.find(otherRank) == interfaceMap.end()) {
                interfaceMap[otherRank].neighborRank = otherRank;
            }
            
            // Add face to interface
            interfaceMap[otherRank].sendFaces.push_back(faceId);
            interfaceMap[otherRank].recvFaces.push_back(faceId);
        }
    }
    
    // Second pass: build cell lists
    for (auto& [rank, interface] : interfaceMap) {
        std::unordered_set<Index> sendCellSet, recvCellSet;
        
        // Collect cells adjacent to interface faces
        for (Index faceId : interface.sendFaces) {
            const Face& face = mesh_.face(faceId);
            
            if (mesh_.cell(face.owner()).processor() == myRank) {
                sendCellSet.insert(face.owner());
            }
            if (!face.isBoundary() && 
                mesh_.cell(face.neighbor()).processor() == myRank) {
                sendCellSet.insert(face.neighbor());
            }
        }
        
        // Convert sets to vectors
        interface.sendCells.assign(sendCellSet.begin(), sendCellSet.end());
        interface.recvCells.assign(recvCellSet.begin(), recvCellSet.end());
        
        // Sort for cache efficiency
        std::sort(interface.sendCells.begin(), interface.sendCells.end());
        std::sort(interface.recvCells.begin(), interface.recvCells.end());
    }
    
    // Convert map to vector
    interfaces_.clear();
    interfaces_.reserve(interfaceMap.size());
    for (const auto& [rank, interface] : interfaceMap) {
        interfaces_.push_back(interface);
    }
}

void CommunicationPattern::buildGhostCellLists() {
    ghostCellMap_.clear();
    
    for (const auto& interface : interfaces_) {
        std::vector<Index>& ghostCells = ghostCellMap_[interface.neighborRank];
        
        // Ghost cells are cells we need from the neighbor
        for (Index cellId : interface.recvCells) {
            const Cell& cell = mesh_.cell(cellId);
            
            // Add cells within stencil radius
            for (const auto& face : cell.faces()) {
                if (!face.isBoundary()) {
                    Index neighborCell = (face.owner() == cellId) ? 
                                       face.neighbor() : face.owner();
                    
                    if (mesh_.cell(neighborCell).processor() != MPIWrapper::rank()) {
                        ghostCells.push_back(neighborCell);
                    }
                }
            }
        }
        
        // Remove duplicates
        std::sort(ghostCells.begin(), ghostCells.end());
        ghostCells.erase(std::unique(ghostCells.begin(), ghostCells.end()), 
                        ghostCells.end());
    }
}

// Global reduction operations
namespace {
    
template<typename T>
T globalReduction(const T& localValue, MPI_Op op) {
    T globalValue = localValue;
    
    if (MPIWrapper::isParallel()) {
        if constexpr (std::is_same_v<T, Real>) {
            MPI_Allreduce(&localValue, &globalValue, 1, 
                         MPIWrapper::getMPIType<Real>(), op, MPI_COMM_WORLD);
        } else if constexpr (std::is_same_v<T, Vector3>) {
            MPI_Allreduce(localValue.data(), globalValue.data(), 3,
                         MPIWrapper::getMPIType<Real>(), op, MPI_COMM_WORLD);
        }
    }
    
    return globalValue;
}

} // anonymous namespace

// Explicit instantiations
template class FieldCommunicator<Real>;
template class FieldCommunicator<Vector3>;

Real globalSum(Real localValue) {
    return globalReduction(localValue, MPI_SUM);
}

Vector3 globalSum(const Vector3& localValue) {
    return globalReduction(localValue, MPI_SUM);
}

Real globalMin(Real localValue) {
    return globalReduction(localValue, MPI_MIN);
}

Real globalMax(Real localValue) {
    return globalReduction(localValue, MPI_MAX);
}

Vector3 globalMin(const Vector3& localValue) {
    return globalReduction(localValue, MPI_MIN);
}

Vector3 globalMax(const Vector3& localValue) {
    return globalReduction(localValue, MPI_MAX);
}

} // namespace cfd::parallel
