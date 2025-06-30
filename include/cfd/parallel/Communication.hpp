#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/parallel/MPI_Wrapper.hpp"
#include <vector>
#include <map>

namespace cfd::parallel {

// Communication pattern for parallel field exchange
class CommunicationPattern {
public:
    struct ProcessorInterface {
        int neighborRank;
        std::vector<Index> sendCells;
        std::vector<Index> recvCells;
        std::vector<Index> sendFaces;
        std::vector<Index> recvFaces;
    };
    
    CommunicationPattern(const Mesh& mesh) : mesh_(mesh) {}
    
    // Build communication pattern
    void build();
    
    // Get interfaces
    const std::vector<ProcessorInterface>& interfaces() const { return interfaces_; }
    
    // Get ghost cells for a processor
    const std::vector<Index>& ghostCells(int rank) const {
        auto it = ghostCellMap_.find(rank);
        return it != ghostCellMap_.end() ? it->second : emptyVector_;
    }
    
private:
    const Mesh& mesh_;
    std::vector<ProcessorInterface> interfaces_;
    std::map<int, std::vector<Index>> ghostCellMap_;
    static const std::vector<Index> emptyVector_;
    
    // Determine processor interfaces
    void findProcessorInterfaces();
    
    // Build ghost cell lists
    void buildGhostCellLists();
};

// Field communicator for parallel data exchange
template<typename T>
class FieldCommunicator {
public:
    FieldCommunicator(const CommunicationPattern& pattern)
        : pattern_(pattern) {}
    
    // Exchange field data between processors
    void exchange(Field<T>& field) {
        const auto& interfaces = pattern_.interfaces();
        
        // Non-blocking sends
        std::vector<MPIWrapper::Request> sendRequests;
        std::vector<std::vector<T>> sendBuffers(interfaces.size());
        
        for (size_t i = 0; i < interfaces.size(); ++i) {
            const auto& interface = interfaces[i];
            auto& buffer = sendBuffers[i];
            
            // Pack send data
            packData(field, interface.sendCells, buffer);
            
            // Send to neighbor
            sendRequests.push_back(
                MPIWrapper::isendVector(buffer, interface.neighborRank, 0)
            );
        }
        
        // Receive and unpack
        for (const auto& interface : interfaces) {
            std::vector<T> recvBuffer;
            MPIWrapper::recvVector(recvBuffer, interface.neighborRank, 0);
            
            // Unpack received data
            unpackData(recvBuffer, interface.recvCells, field);
        }
        
        // Wait for sends to complete
        MPIWrapper::waitAll(sendRequests);
    }
    
    // Exchange with custom packing/unpacking
    void exchange(Field<T>& field,
                 std::function<void(const Field<T>&, const std::vector<Index>&, std::vector<T>&)> pack,
                 std::function<void(const std::vector<T>&, const std::vector<Index>&, Field<T>&)> unpack) {
        const auto& interfaces = pattern_.interfaces();
        
        std::vector<MPIWrapper::Request> sendRequests;
        std::vector<std::vector<T>> sendBuffers(interfaces.size());
        
        for (size_t i = 0; i < interfaces.size(); ++i) {
            const auto& interface = interfaces[i];
            pack(field, interface.sendCells, sendBuffers[i]);
            sendRequests.push_back(
                MPIWrapper::isendVector(sendBuffers[i], interface.neighborRank, 0)
            );
        }
        
        for (const auto& interface : interfaces) {
            std::vector<T> recvBuffer;
            MPIWrapper::recvVector(recvBuffer, interface.neighborRank, 0);
            unpack(recvBuffer, interface.recvCells, field);
        }
        
        MPIWrapper::waitAll(sendRequests);
    }
    
private:
    const CommunicationPattern& pattern_;
    
    // Default packing
    void packData(const Field<T>& field, const std::vector<Index>& cells,
                  std::vector<T>& buffer) {
        buffer.resize(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            buffer[i] = field[cells[i]];
        }
    }
    
    // Default unpacking
    void unpackData(const std::vector<T>& buffer, const std::vector<Index>& cells,
                    Field<T>& field) {
        for (size_t i = 0; i < cells.size(); ++i) {
            field[cells[i]] = buffer[i];
        }
    }
};

// Global field operations
class GlobalFieldOps {
public:
    // Global sum of field
    template<typename T>
    static T globalSum(const Field<T>& field) {
        T localSum = T();
        for (Index i = 0; i < field.mesh().numCells(); ++i) {
            if constexpr (std::is_same_v<T, Real>) {
                localSum += field[i] * field.mesh().cell(i).volume();
            } else if constexpr (std::is_same_v<T, Vector3>) {
                localSum += field[i] * field.mesh().cell(i).volume();
            }
        }
        return parallel::globalSum(localSum);
    }
    
    // Global min/max
    template<typename T>
    static T globalMin(const Field<T>& field) {
        T localMin = field[0];
        for (Index i = 1; i < field.mesh().numCells(); ++i) {
            if constexpr (std::is_same_v<T, Real>) {
                localMin = std::min(localMin, field[i]);
            } else if constexpr (std::is_same_v<T, Vector3>) {
                for (int j = 0; j < 3; ++j) {
                    localMin[j] = std::min(localMin[j], field[i][j]);
                }
            }
        }
        return parallel::globalMin(localMin);
    }
    
    template<typename T>
    static T globalMax(const Field<T>& field) {
        T localMax = field[0];
        for (Index i = 1; i < field.mesh().numCells(); ++i) {
            if constexpr (std::is_same_v<T, Real>) {
                localMax = std::max(localMax, field[i]);
            } else if constexpr (std::is_same_v<T, Vector3>) {
                for (int j = 0; j < 3; ++j) {
                    localMax[j] = std::max(localMax[j], field[i][j]);
                }
            }
        }
        return parallel::globalMax(localMax);
    }
    
    // Global L2 norm
    static Real globalL2Norm(const ScalarField& field) {
        Real localSum = 0.0;
        for (Index i = 0; i < field.mesh().numCells(); ++i) {
            Real vol = field.mesh().cell(i).volume();
            localSum += field[i] * field[i] * vol;
        }
        return std::sqrt(parallel::globalSum(localSum));
    }
    
    static Real globalL2Norm(const VectorField& field) {
        Real localSum = 0.0;
        for (Index i = 0; i < field.mesh().numCells(); ++i) {
            Real vol = field.mesh().cell(i).volume();
            localSum += field[i].squaredNorm() * vol;
        }
        return std::sqrt(parallel::globalSum(localSum));
    }
};

// Parallel matrix operations
class ParallelMatrixOps {
public:
    // Parallel matrix-vector multiplication
    static void multiply(const SparseMatrix& A, const VectorX& x, VectorX& y,
                        const CommunicationPattern& pattern) {
        // Local multiplication
        y = A * x;
        
        // Exchange ghost values if needed
        // This would require more sophisticated handling of distributed matrices
    }
    
    // Parallel dot product
    static Real dot(const VectorX& x, const VectorX& y) {
        Real localDot = x.dot(y);
        return parallel::globalSum(localDot);
    }
    
    // Parallel norm
    static Real norm(const VectorX& x) {
        Real localNorm2 = x.squaredNorm();
        return std::sqrt(parallel::globalSum(localNorm2));
    }
};

// Load balancing utilities
class LoadBalancer {
public:
    struct LoadInfo {
        int rank;
        Real load;
        Index numCells;
        Index numFaces;
    };
    
    // Compute load imbalance
    static Real computeImbalance(const std::vector<LoadInfo>& loads) {
        Real avgLoad = 0.0;
        for (const auto& info : loads) {
            avgLoad += info.load;
        }
        avgLoad /= loads.size();
        
        Real maxLoad = 0.0;
        for (const auto& info : loads) {
            maxLoad = std::max(maxLoad, info.load);
        }
        
        return maxLoad / avgLoad - 1.0;
    }
    
    // Dynamic load balancing (simplified)
    static std::vector<Index> rebalance(const std::vector<LoadInfo>& loads,
                                       const std::vector<Index>& currentPartition) {
        // This is a placeholder for sophisticated load balancing algorithms
        // like recursive coordinate bisection, graph partitioning, etc.
        return currentPartition;
    }
};

// Parallel I/O coordinator
class ParallelIO {
public:
    // Collective write
    template<typename T>
    static void collectiveWrite(const std::string& filename,
                               const Field<T>& field,
                               const CommunicationPattern& pattern) {
        if (MPIWrapper::isMaster()) {
            // Master collects and writes
            std::vector<T> globalData;
            gatherField(field, globalData, pattern);
            
            // Write to file
            std::ofstream file(filename, std::ios::binary);
            file.write(reinterpret_cast<const char*>(globalData.data()),
                      globalData.size() * sizeof(T));
        } else {
            // Workers send their data
            sendFieldToMaster(field, pattern);
        }
    }
    
    // Collective read
    template<typename T>
    static void collectiveRead(const std::string& filename,
                              Field<T>& field,
                              const CommunicationPattern& pattern) {
        if (MPIWrapper::isMaster()) {
            // Master reads and distributes
            std::vector<T> globalData;
            std::ifstream file(filename, std::ios::binary);
            // Read data...
            
            scatterField(globalData, field, pattern);
        } else {
            // Workers receive their data
            receiveFieldFromMaster(field, pattern);
        }
    }
    
private:
    template<typename T>
    static void gatherField(const Field<T>& field,
                           std::vector<T>& globalData,
                           const CommunicationPattern& pattern) {
        // Implementation of gathering field data
    }
    
    template<typename T>
    static void scatterField(const std::vector<T>& globalData,
                            Field<T>& field,
                            const CommunicationPattern& pattern) {
        // Implementation of scattering field data
    }
    
    template<typename T>
    static void sendFieldToMaster(const Field<T>& field,
                                 const CommunicationPattern& pattern) {
        // Implementation of sending to master
    }
    
    template<typename T>
    static void receiveFieldFromMaster(Field<T>& field,
                                      const CommunicationPattern& pattern) {
        // Implementation of receiving from master
    }
};

} // namespace cfd::parallel