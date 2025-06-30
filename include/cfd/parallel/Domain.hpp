#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Mesh.hpp"
#include "cfd/parallel/MPI_Wrapper.hpp"
#include <vector>
#include <map>
#include <set>

namespace cfd::parallel {

// Domain decomposition for parallel computing
class DomainDecomposition {
public:
    struct Partition {
        std::vector<Index> cells;           // Local cells
        std::vector<Index> ghostCells;      // Ghost cells from other processors
        std::map<int, std::vector<Index>> sharedCells;  // Cells shared with each processor
        std::map<int, std::vector<Index>> sharedFaces;  // Faces shared with each processor
    };
    
    DomainDecomposition(const Mesh& globalMesh, int numPartitions)
        : globalMesh_(globalMesh), numPartitions_(numPartitions) {}
    
    // Decomposition methods
    enum class Method {
        SIMPLE,           // Simple geometric partitioning
        METIS,           // METIS graph partitioning
        PARMETIS,        // Parallel METIS
        SCOTCH,          // SCOTCH partitioning
        ZOLTAN,          // Zoltan partitioning
        COORDINATE,      // Recursive coordinate bisection
        HILBERT          // Hilbert space-filling curve
    };
    
    // Perform decomposition
    void decompose(Method method = Method::METIS);
    
    // Get partition for a rank
    const Partition& getPartition(int rank) const {
        return partitions_[rank];
    }
    
    // Get cell partition mapping
    const std::vector<int>& getCellPartition() const {
        return cellPartition_;
    }
    
    // Create local mesh for a partition
    SharedPtr<Mesh> createLocalMesh(int rank) const;
    
    // Compute decomposition quality metrics
    struct QualityMetrics {
        Real loadImbalance;      // Max/avg - 1
        Real communicationVolume; // Total shared faces
        Real edgeCut;            // Number of cut edges
        int maxNeighbors;        // Maximum neighbors per partition
    };
    
    QualityMetrics computeQuality() const;
    
private:
    const Mesh& globalMesh_;
    int numPartitions_;
    std::vector<int> cellPartition_;  // Partition ID for each cell
    std::vector<Partition> partitions_;
    
    // Decomposition implementations
    void decomposeSimple();
    void decomposeWithMETIS();
    void decomposeWithCoordinateBisection();
    void decomposeWithHilbert();
    
    // Build partition data structures
    void buildPartitions();
    void identifyGhostCells();
    void identifySharedEntities();
    
    // Graph construction for partitioning
    struct Graph {
        std::vector<int> xadj;     // Adjacency index array
        std::vector<int> adjncy;   // Adjacency array
        std::vector<int> vwgt;     // Vertex weights
        std::vector<int> adjwgt;   // Edge weights
    };
    
    Graph buildDualGraph() const;
    
    // Coordinate bisection helper
    void recursiveBisection(const std::vector<Index>& cells,
                           std::vector<int>& partition,
                           int level, int maxLevel);
    
    // Hilbert curve helper
    uint64_t hilbertIndex(const Vector3& point, int level) const;
};

// Local mesh manager for parallel execution
class LocalMeshManager {
public:
    LocalMeshManager(SharedPtr<Mesh> localMesh,
                    const DomainDecomposition::Partition& partition)
        : localMesh_(localMesh), partition_(partition) {
        buildMaps();
    }
    
    // Access local mesh
    Mesh& mesh() { return *localMesh_; }
    const Mesh& mesh() const { return *localMesh_; }
    
    // Mapping functions
    Index globalToLocal(Index globalId) const {
        auto it = globalToLocalMap_.find(globalId);
        return it != globalToLocalMap_.end() ? it->second : -1;
    }
    
    Index localToGlobal(Index localId) const {
        return localToGlobalMap_[localId];
    }
    
    // Check if cell is ghost
    bool isGhostCell(Index localId) const {
        return ghostCellSet_.count(localId) > 0;
    }
    
    // Get processor interfaces
    const std::map<int, std::vector<Index>>& processorInterfaces() const {
        return processorInterfaces_;
    }
    
private:
    SharedPtr<Mesh> localMesh_;
    DomainDecomposition::Partition partition_;
    
    std::vector<Index> localToGlobalMap_;
    std::map<Index, Index> globalToLocalMap_;
    std::set<Index> ghostCellSet_;
    std::map<int, std::vector<Index>> processorInterfaces_;
    
    void buildMaps();
};

// Parallel mesh redistribution
class MeshRedistribution {
public:
    // Redistribute mesh based on new partitioning
    static void redistribute(Mesh& mesh,
                           const std::vector<int>& oldPartition,
                           const std::vector<int>& newPartition);
    
    // Dynamic load balancing
    static void dynamicLoadBalance(Mesh& mesh,
                                  const std::vector<Real>& cellWeights);
    
    // Adaptive mesh redistribution
    static void adaptiveRedistribute(Mesh& mesh,
                                   const std::vector<bool>& refinementFlags);
};

// Halo exchange pattern
class HaloExchange {
public:
    HaloExchange(const LocalMeshManager& meshManager)
        : meshManager_(meshManager) {
        buildExchangePattern();
    }
    
    // Exchange halo data
    template<typename T>
    void exchange(std::vector<T>& data) {
        // Send data
        std::vector<MPIWrapper::Request> sendRequests;
        std::map<int, std::vector<T>> sendBuffers;
        
        for (const auto& [rank, cells] : sendPattern_) {
            auto& buffer = sendBuffers[rank];
            buffer.reserve(cells.size());
            
            for (Index cell : cells) {
                buffer.push_back(data[cell]);
            }
            
            sendRequests.push_back(
                MPIWrapper::isendVector(buffer, rank, 0)
            );
        }
        
        // Receive data
        for (const auto& [rank, cells] : recvPattern_) {
            std::vector<T> buffer;
            MPIWrapper::recvVector(buffer, rank, 0);
            
            for (size_t i = 0; i < cells.size(); ++i) {
                data[cells[i]] = buffer[i];
            }
        }
        
        // Wait for sends
        MPIWrapper::waitAll(sendRequests);
    }
    
    // Exchange with custom packing
    template<typename T, typename PackFunc, typename UnpackFunc>
    void exchange(T& data, PackFunc pack, UnpackFunc unpack) {
        std::vector<MPIWrapper::Request> sendRequests;
        std::map<int, std::vector<char>> sendBuffers;
        
        for (const auto& [rank, cells] : sendPattern_) {
            auto& buffer = sendBuffers[rank];
            pack(data, cells, buffer);
            
            sendRequests.push_back(
                MPIWrapper::isendVector(buffer, rank, 0)
            );
        }
        
        for (const auto& [rank, cells] : recvPattern_) {
            std::vector<char> buffer;
            MPIWrapper::recvVector(buffer, rank, 0);
            unpack(buffer, cells, data);
        }
        
        MPIWrapper::waitAll(sendRequests);
    }
    
private:
    const LocalMeshManager& meshManager_;
    std::map<int, std::vector<Index>> sendPattern_;
    std::map<int, std::vector<Index>> recvPattern_;
    
    void buildExchangePattern();
};

// Parallel mesh partitioner interface
class MeshPartitioner {
public:
    virtual ~MeshPartitioner() = default;
    
    // Partition mesh
    virtual std::vector<int> partition(const Mesh& mesh, int numParts) = 0;
    
    // Repartition with constraints
    virtual std::vector<int> repartition(const Mesh& mesh,
                                        const std::vector<int>& currentPartition,
                                        const std::vector<Real>& weights,
                                        int numParts) = 0;
};

// METIS partitioner implementation
#ifdef CFD_USE_METIS
class METISPartitioner : public MeshPartitioner {
public:
    struct Options {
        int ncon = 1;           // Number of constraints
        int niter = 10;         // Number of iterations
        int seed = 0;           // Random seed
        Real ubvec = 1.05;      // Load imbalance tolerance
        bool minimize_comm = true;  // Minimize communication volume
    };
    
    METISPartitioner(const Options& options = Options())
        : options_(options) {}
    
    std::vector<int> partition(const Mesh& mesh, int numParts) override;
    
    std::vector<int> repartition(const Mesh& mesh,
                                const std::vector<int>& currentPartition,
                                const std::vector<Real>& weights,
                                int numParts) override;
    
private:
    Options options_;
};
#endif

// Zoltan partitioner implementation
#ifdef CFD_USE_ZOLTAN
class ZoltanPartitioner : public MeshPartitioner {
public:
    struct Options {
        std::string method = "RCB";  // RCB, RIB, HSFC, GRAPH, HYPERGRAPH
        Real imbalance_tol = 1.05;
        int debug_level = 0;
    };
    
    ZoltanPartitioner(const Options& options = Options())
        : options_(options) {}
    
    std::vector<int> partition(const Mesh& mesh, int numParts) override;
    
    std::vector<int> repartition(const Mesh& mesh,
                                const std::vector<int>& currentPartition,
                                const std::vector<Real>& weights,
                                int numParts) override;
    
private:
    Options options_;
};
#endif

// Factory for creating partitioners
inline SharedPtr<MeshPartitioner> createPartitioner(const std::string& type) {
    #ifdef CFD_USE_METIS
    if (type == "METIS") {
        return std::make_shared<METISPartitioner>();
    }
    #endif
    
    #ifdef CFD_USE_ZOLTAN
    if (type == "Zoltan") {
        return std::make_shared<ZoltanPartitioner>();
    }
    #endif
    
    throw std::runtime_error("Partitioner type not available: " + type);
}

} // namespace cfd::parallel