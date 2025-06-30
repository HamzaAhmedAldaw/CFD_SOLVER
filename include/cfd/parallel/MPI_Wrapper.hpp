#pragma once

#include "cfd/core/Types.hpp"
#ifdef CFD_ENABLE_MPI
#include <mpi.h>
#endif
#include <vector>
#include <memory>

namespace cfd::parallel {

// MPI wrapper for parallel computing
class MPIWrapper {
public:
    // Initialize MPI
    static void initialize(int& argc, char**& argv) {
        #ifdef CFD_ENABLE_MPI
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        initialized_ = true;
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        #else
        rank_ = 0;
        size_ = 1;
        #endif
    }
    
    // Finalize MPI
    static void finalize() {
        #ifdef CFD_ENABLE_MPI
        if (initialized_) {
            MPI_Finalize();
            initialized_ = false;
        }
        #endif
    }
    
    // Get rank and size
    static int rank() { return rank_; }
    static int size() { return size_; }
    static bool isParallel() { return size_ > 1; }
    static bool isMaster() { return rank_ == 0; }
    
    // Basic communication operations
    template<typename T>
    static void send(const T& data, int dest, int tag = 0) {
        #ifdef CFD_ENABLE_MPI
        MPI_Send(&data, sizeof(T), MPI_BYTE, dest, tag, MPI_COMM_WORLD);
        #endif
    }
    
    template<typename T>
    static void recv(T& data, int source, int tag = 0) {
        #ifdef CFD_ENABLE_MPI
        MPI_Status status;
        MPI_Recv(&data, sizeof(T), MPI_BYTE, source, tag, MPI_COMM_WORLD, &status);
        #endif
    }
    
    // Vector operations
    template<typename T>
    static void sendVector(const std::vector<T>& vec, int dest, int tag = 0) {
        #ifdef CFD_ENABLE_MPI
        int size = vec.size();
        MPI_Send(&size, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        if (size > 0) {
            MPI_Send(vec.data(), size * sizeof(T), MPI_BYTE, dest, tag + 1, MPI_COMM_WORLD);
        }
        #endif
    }
    
    template<typename T>
    static void recvVector(std::vector<T>& vec, int source, int tag = 0) {
        #ifdef CFD_ENABLE_MPI
        int size;
        MPI_Status status;
        MPI_Recv(&size, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        vec.resize(size);
        if (size > 0) {
            MPI_Recv(vec.data(), size * sizeof(T), MPI_BYTE, source, tag + 1, 
                     MPI_COMM_WORLD, &status);
        }
        #endif
    }
    
    // Collective operations
    template<typename T>
    static void broadcast(T& data, int root = 0) {
        #ifdef CFD_ENABLE_MPI
        MPI_Bcast(&data, sizeof(T), MPI_BYTE, root, MPI_COMM_WORLD);
        #endif
    }
    
    template<typename T>
    static void allReduce(T& value, MPI_Op op = MPI_SUM) {
        #ifdef CFD_ENABLE_MPI
        T result;
        MPI_Allreduce(&value, &result, 1, getMPIType<T>(), op, MPI_COMM_WORLD);
        value = result;
        #endif
    }
    
    template<typename T>
    static void gather(const T& sendData, std::vector<T>& recvData, int root = 0) {
        #ifdef CFD_ENABLE_MPI
        if (rank_ == root) {
            recvData.resize(size_);
        }
        MPI_Gather(&sendData, sizeof(T), MPI_BYTE,
                   recvData.data(), sizeof(T), MPI_BYTE,
                   root, MPI_COMM_WORLD);
        #else
        recvData.resize(1);
        recvData[0] = sendData;
        #endif
    }
    
    template<typename T>
    static void allGather(const T& sendData, std::vector<T>& recvData) {
        #ifdef CFD_ENABLE_MPI
        recvData.resize(size_);
        MPI_Allgather(&sendData, sizeof(T), MPI_BYTE,
                      recvData.data(), sizeof(T), MPI_BYTE,
                      MPI_COMM_WORLD);
        #else
        recvData.resize(1);
        recvData[0] = sendData;
        #endif
    }
    
    // Barrier synchronization
    static void barrier() {
        #ifdef CFD_ENABLE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
    }
    
    // Non-blocking operations
    class Request {
    public:
        Request() = default;
        
        void wait() {
            #ifdef CFD_ENABLE_MPI
            if (request_ != MPI_REQUEST_NULL) {
                MPI_Status status;
                MPI_Wait(&request_, &status);
            }
            #endif
        }
        
        bool test() {
            #ifdef CFD_ENABLE_MPI
            if (request_ != MPI_REQUEST_NULL) {
                int flag;
                MPI_Status status;
                MPI_Test(&request_, &flag, &status);
                return flag != 0;
            }
            #endif
            return true;
        }
        
    private:
        #ifdef CFD_ENABLE_MPI
        MPI_Request request_ = MPI_REQUEST_NULL;
        #endif
        
        friend class MPIWrapper;
    };
    
    template<typename T>
    static Request isend(const T& data, int dest, int tag = 0) {
        Request req;
        #ifdef CFD_ENABLE_MPI
        MPI_Isend(&data, sizeof(T), MPI_BYTE, dest, tag, 
                  MPI_COMM_WORLD, &req.request_);
        #endif
        return req;
    }
    
    template<typename T>
    static Request irecv(T& data, int source, int tag = 0) {
        Request req;
        #ifdef CFD_ENABLE_MPI
        MPI_Irecv(&data, sizeof(T), MPI_BYTE, source, tag,
                  MPI_COMM_WORLD, &req.request_);
        #endif
        return req;
    }
    
    // Wait for multiple requests
    static void waitAll(std::vector<Request>& requests) {
        #ifdef CFD_ENABLE_MPI
        std::vector<MPI_Request> mpiRequests;
        for (auto& req : requests) {
            mpiRequests.push_back(req.request_);
        }
        MPI_Waitall(mpiRequests.size(), mpiRequests.data(), MPI_STATUSES_IGNORE);
        #endif
    }
    
private:
    static int rank_;
    static int size_;
    static bool initialized_;
    
    // MPI datatype mapping
    template<typename T>
    static MPI_Datatype getMPIType() {
        #ifdef CFD_ENABLE_MPI
        if constexpr (std::is_same_v<T, int>) return MPI_INT;
        else if constexpr (std::is_same_v<T, float>) return MPI_FLOAT;
        else if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
        else return MPI_BYTE;
        #else
        return 0;
        #endif
    }
};

// Static member initialization
inline int MPIWrapper::rank_ = 0;
inline int MPIWrapper::size_ = 1;
inline bool MPIWrapper::initialized_ = false;

// RAII MPI initialization
class MPIEnvironment {
public:
    MPIEnvironment(int& argc, char**& argv) {
        MPIWrapper::initialize(argc, argv);
    }
    
    ~MPIEnvironment() {
        MPIWrapper::finalize();
    }
    
    // Delete copy and move
    MPIEnvironment(const MPIEnvironment&) = delete;
    MPIEnvironment& operator=(const MPIEnvironment&) = delete;
    MPIEnvironment(MPIEnvironment&&) = delete;
    MPIEnvironment& operator=(MPIEnvironment&&) = delete;
};

// Global reduction operations
template<typename T>
T globalSum(const T& localValue) {
    T result = localValue;
    MPIWrapper::allReduce(result, MPI_SUM);
    return result;
}

template<typename T>
T globalMin(const T& localValue) {
    T result = localValue;
    MPIWrapper::allReduce(result, MPI_MIN);
    return result;
}

template<typename T>
T globalMax(const T& localValue) {
    T result = localValue;
    MPIWrapper::allReduce(result, MPI_MAX);
    return result;
}

// Parallel timer
class ParallelTimer {
public:
    ParallelTimer(const std::string& name) : name_(name) {
        start_ = MPI_Wtime();
    }
    
    ~ParallelTimer() {
        Real localTime = MPI_Wtime() - start_;
        Real maxTime = globalMax(localTime);
        Real minTime = globalMin(localTime);
        Real avgTime = globalSum(localTime) / MPIWrapper::size();
        
        if (MPIWrapper::isMaster()) {
            std::cout << "Timer " << name_ << ": "
                     << "avg=" << avgTime << "s, "
                     << "min=" << minTime << "s, "
                     << "max=" << maxTime << "s\n";
        }
    }
    
private:
    std::string name_;
    Real start_;
};

} // namespace cfd::parallel