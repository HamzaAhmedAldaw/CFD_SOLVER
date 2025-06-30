// === src/memory/MemoryPool.cpp ===
#include "cfd/memory/MemoryPool.hpp"
#include <algorithm>
#include <cstring>

namespace cfd::memory {

MemoryPool::MemoryPool(size_t blockSize, size_t blocksPerChunk)
    : blockSize_(blockSize), blocksPerChunk_(blocksPerChunk),
      totalAllocated_(0), totalDeallocated_(0) {
    
    // Ensure proper alignment
    blockSize_ = ((blockSize_ + alignof(std::max_align_t) - 1) / 
                  alignof(std::max_align_t)) * alignof(std::max_align_t);
    
    // Allocate first chunk
    allocateChunk();
}

MemoryPool::~MemoryPool() {
    // All chunks are automatically freed by unique_ptr
}

void* MemoryPool::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (freeList_.empty()) {
        allocateChunk();
    }
    
    void* ptr = freeList_.back();
    freeList_.pop_back();
    
    totalAllocated_++;
    
    return ptr;
}

void MemoryPool::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if pointer belongs to this pool
    bool found = false;
    for (const auto& chunk : chunks_) {
        char* start = chunk.get();
        char* end = start + blockSize_ * blocksPerChunk_;
        
        if (ptr >= start && ptr < end) {
            found = true;
            break;
        }
    }
    
    if (!found) {
        throw std::runtime_error("Pointer does not belong to this memory pool");
    }
    
    freeList_.push_back(ptr);
    totalDeallocated_++;
}

void MemoryPool::allocateChunk() {
    // Allocate aligned memory for chunk
    size_t chunkSize = blockSize_ * blocksPerChunk_;
    auto chunk = std::make_unique<char[]>(chunkSize);
    
    char* start = chunk.get();
    
    // Add blocks to free list
    for (size_t i = 0; i < blocksPerChunk_; ++i) {
        freeList_.push_back(start + i * blockSize_);
    }
    
    chunks_.push_back(std::move(chunk));
}

size_t MemoryPool::getMemoryUsage() const {
    return chunks_.size() * blockSize_ * blocksPerChunk_;
}

// Thread-local memory pools
thread_local std::unique_ptr<MemoryPool> tlsSmallPool;
thread_local std::unique_ptr<MemoryPool> tlsMediumPool;
thread_local std::unique_ptr<MemoryPool> tlsLargePool;

void* allocateFromPool(size_t size) {
    // Initialize pools if needed
    if (!tlsSmallPool) {
        tlsSmallPool = std::make_unique<MemoryPool>(64, 1024);
        tlsMediumPool = std::make_unique<MemoryPool>(512, 256);
        tlsLargePool = std::make_unique<MemoryPool>(4096, 64);
    }
    
    // Select appropriate pool
    if (size <= 64) {
        return tlsSmallPool->allocate();
    } else if (size <= 512) {
        return tlsMediumPool->allocate();
    } else if (size <= 4096) {
        return tlsLargePool->allocate();
    } else {
        // Fall back to regular allocation for very large blocks
        return std::aligned_alloc(alignof(std::max_align_t), size);
    }
}

void deallocateToPool(void* ptr, size_t size) {
    if (!ptr) return;
    
    // Determine which pool to use
    if (size <= 64 && tlsSmallPool) {
        tlsSmallPool->deallocate(ptr);
    } else if (size <= 512 && tlsMediumPool) {
        tlsMediumPool->deallocate(ptr);
    } else if (size <= 4096 && tlsLargePool) {
        tlsLargePool->deallocate(ptr);
    } else {
        // Regular deallocation
        std::free(ptr);
    }
}

// Object pool implementation
template<typename T>
ObjectPool<T>::ObjectPool(size_t initialSize) {
    pool_.reserve(initialSize);
    
    for (size_t i = 0; i < initialSize; ++i) {
        pool_.emplace_back(std::make_unique<T>());
    }
}

template<typename T>
template<typename... Args>
T* ObjectPool<T>::acquire(Args&&... args) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (pool_.empty()) {
        // Create new object
        return new T(std::forward<Args>(args)...);
    }
    
    // Get object from pool
    std::unique_ptr<T> obj = std::move(pool_.back());
    pool_.pop_back();
    
    // Reinitialize object
    *obj = T(std::forward<Args>(args)...);
    
    return obj.release();
}

template<typename T>
void ObjectPool<T>::release(T* obj) {
    if (!obj) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Reset object to default state
    *obj = T();
    
    // Return to pool
    pool_.emplace_back(obj);
}

// Explicit instantiations
template class ObjectPool<Cell>;
template class ObjectPool<Face>;

} // namespace cfd::memory
