#pragma once

#include "cfd/core/Types.hpp"
#include <vector>
#include <stack>
#include <memory>
#include <mutex>
#include <cstddef>
#include <algorithm>

namespace cfd::memory {

// Memory pool for efficient allocation/deallocation
class MemoryPool {
public:
    // Constructor with block size and initial capacity
    MemoryPool(size_t blockSize, size_t initialBlocks = 1024)
        : blockSize_(alignSize(blockSize)), 
          blocksPerChunk_(initialBlocks) {
        allocateNewChunk();
    }
    
    ~MemoryPool() {
        for (auto& chunk : chunks_) {
            std::free(chunk.memory);
        }
    }
    
    // Allocate a block
    void* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (freeBlocks_.empty()) {
            allocateNewChunk();
        }
        
        void* block = freeBlocks_.top();
        freeBlocks_.pop();
        ++allocatedBlocks_;
        
        return block;
    }
    
    // Deallocate a block
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Verify pointer belongs to this pool
        if (!ownsPointer(ptr)) {
            throw std::runtime_error("Pointer does not belong to this memory pool");
        }
        
        freeBlocks_.push(ptr);
        --allocatedBlocks_;
    }
    
    // Get statistics
    size_t blockSize() const { return blockSize_; }
    size_t allocatedBlocks() const { return allocatedBlocks_; }
    size_t totalBlocks() const { return totalBlocks_; }
    size_t usedMemory() const { return allocatedBlocks_ * blockSize_; }
    size_t totalMemory() const { return totalBlocks_ * blockSize_; }
    
    // Reset pool (deallocate all)
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Clear free list and rebuild
        while (!freeBlocks_.empty()) {
            freeBlocks_.pop();
        }
        
        // Add all blocks back to free list
        for (auto& chunk : chunks_) {
            char* block = static_cast<char*>(chunk.memory);
            for (size_t i = 0; i < chunk.numBlocks; ++i) {
                freeBlocks_.push(block + i * blockSize_);
            }
        }
        
        allocatedBlocks_ = 0;
    }
    
private:
    struct Chunk {
        void* memory;
        size_t numBlocks;
    };
    
    size_t blockSize_;
    size_t blocksPerChunk_;
    size_t allocatedBlocks_ = 0;
    size_t totalBlocks_ = 0;
    
    std::vector<Chunk> chunks_;
    std::stack<void*> freeBlocks_;
    mutable std::mutex mutex_;
    
    // Align size to cache line
    static size_t alignSize(size_t size) {
        const size_t alignment = CACHE_LINE_SIZE;
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
    // Allocate new chunk
    void allocateNewChunk() {
        size_t chunkSize = blockSize_ * blocksPerChunk_;
        void* memory = std::aligned_alloc(CACHE_LINE_SIZE, chunkSize);
        
        if (!memory) {
            throw std::bad_alloc();
        }
        
        chunks_.push_back({memory, blocksPerChunk_});
        
        // Add blocks to free list
        char* block = static_cast<char*>(memory);
        for (size_t i = 0; i < blocksPerChunk_; ++i) {
            freeBlocks_.push(block + i * blockSize_);
        }
        
        totalBlocks_ += blocksPerChunk_;
        
        // Double chunk size for next allocation (geometric growth)
        blocksPerChunk_ *= 2;
    }
    
    // Check if pointer belongs to this pool
    bool ownsPointer(void* ptr) const {
        for (const auto& chunk : chunks_) {
            char* start = static_cast<char*>(chunk.memory);
            char* end = start + chunk.numBlocks * blockSize_;
            char* p = static_cast<char*>(ptr);
            
            if (p >= start && p < end) {
                // Check alignment
                return (p - start) % blockSize_ == 0;
            }
        }
        return false;
    }
};

// Object pool for typed allocations
template<typename T>
class ObjectPool {
public:
    ObjectPool(size_t initialCapacity = 1024)
        : pool_(sizeof(T), initialCapacity) {}
    
    // Allocate and construct object
    template<typename... Args>
    T* construct(Args&&... args) {
        void* memory = pool_.allocate();
        return new(memory) T(std::forward<Args>(args)...);
    }
    
    // Destroy and deallocate object
    void destroy(T* obj) {
        if (!obj) return;
        obj->~T();
        pool_.deallocate(obj);
    }
    
    // STL-compatible allocator interface
    class Allocator {
    public:
        using value_type = T;
        
        Allocator(ObjectPool* pool) : pool_(pool) {}
        
        template<typename U>
        Allocator(const Allocator<U>& other) : pool_(other.pool_) {}
        
        T* allocate(size_t n) {
            if (n != 1) {
                throw std::runtime_error("ObjectPool allocator only supports single allocations");
            }
            return static_cast<T*>(pool_->pool_.allocate());
        }
        
        void deallocate(T* ptr, size_t n) {
            if (n != 1) return;
            pool_->pool_.deallocate(ptr);
        }
        
        template<typename U, typename... Args>
        void construct(U* ptr, Args&&... args) {
            new(ptr) U(std::forward<Args>(args)...);
        }
        
        template<typename U>
        void destroy(U* ptr) {
            ptr->~U();
        }
        
    private:
        ObjectPool* pool_;
        
        template<typename U>
        friend class Allocator;
    };
    
    Allocator getAllocator() { return Allocator(this); }
    
private:
    MemoryPool pool_;
};

// Arena allocator for temporary allocations
class ArenaAllocator {
public:
    ArenaAllocator(size_t size = 1024 * 1024)  // 1MB default
        : size_(size), used_(0) {
        memory_ = static_cast<char*>(std::aligned_alloc(SIMD_ALIGNMENT, size));
        if (!memory_) {
            throw std::bad_alloc();
        }
    }
    
    ~ArenaAllocator() {
        std::free(memory_);
    }
    
    // Allocate memory
    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        // Align current position
        size_t aligned = (used_ + alignment - 1) & ~(alignment - 1);
        
        if (aligned + size > size_) {
            throw std::bad_alloc();
        }
        
        void* ptr = memory_ + aligned;
        used_ = aligned + size;
        
        return ptr;
    }
    
    // Reset arena (deallocate all)
    void reset() {
        used_ = 0;
    }
    
    // Get usage statistics
    size_t used() const { return used_; }
    size_t available() const { return size_ - used_; }
    size_t capacity() const { return size_; }
    
private:
    char* memory_;
    size_t size_;
    size_t used_;
};

// Stack allocator with markers
class StackAllocator {
public:
    using Marker = size_t;
    
    StackAllocator(size_t size) : arena_(size) {}
    
    // Allocate memory
    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        return arena_.allocate(size, alignment);
    }
    
    // Get current position marker
    Marker getMarker() const {
        return arena_.used();
    }
    
    // Roll back to marker
    void rollback(Marker marker) {
        if (marker <= arena_.capacity()) {
            arena_.reset();
            // Note: Simple implementation - just reset everything
            // A proper implementation would track the exact position
        }
    }
    
    // Reset allocator
    void reset() {
        arena_.reset();
    }
    
private:
    ArenaAllocator arena_;
};

// Memory statistics tracker
class MemoryTracker {
public:
    static MemoryTracker& instance() {
        static MemoryTracker tracker;
        return tracker;
    }
    
    void recordAllocation(size_t size, const std::string& category = "default") {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_[category].allocations++;
        stats_[category].totalAllocated += size;
        stats_[category].currentAllocated += size;
        stats_[category].peakAllocated = std::max(stats_[category].peakAllocated,
                                                  stats_[category].currentAllocated);
    }
    
    void recordDeallocation(size_t size, const std::string& category = "default") {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_[category].deallocations++;
        stats_[category].currentAllocated -= size;
    }
    
    struct Stats {
        size_t allocations = 0;
        size_t deallocations = 0;
        size_t totalAllocated = 0;
        size_t currentAllocated = 0;
        size_t peakAllocated = 0;
    };
    
    Stats getStats(const std::string& category = "default") const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stats_.find(category);
        return it != stats_.end() ? it->second : Stats{};
    }
    
    void printReport() const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [category, stats] : stats_) {
            std::cout << "Memory Category: " << category << "\n";
            std::cout << "  Allocations: " << stats.allocations << "\n";
            std::cout << "  Deallocations: " << stats.deallocations << "\n";
            std::cout << "  Total Allocated: " << stats.totalAllocated / (1024.0 * 1024.0) << " MB\n";
            std::cout << "  Current Allocated: " << stats.currentAllocated / (1024.0 * 1024.0) << " MB\n";
            std::cout << "  Peak Allocated: " << stats.peakAllocated / (1024.0 * 1024.0) << " MB\n\n";
        }
    }
    
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Stats> stats_;
    
    MemoryTracker() = default;
};

// RAII wrapper for pool allocations
template<typename T>
class PoolPtr {
public:
    PoolPtr() : ptr_(nullptr), pool_(nullptr) {}
    
    PoolPtr(T* ptr, ObjectPool<T>* pool) 
        : ptr_(ptr), pool_(pool) {}
    
    ~PoolPtr() {
        if (ptr_ && pool_) {
            pool_->destroy(ptr_);
        }
    }
    
    // Move semantics
    PoolPtr(PoolPtr&& other) noexcept
        : ptr_(other.ptr_), pool_(other.pool_) {
        other.ptr_ = nullptr;
        other.pool_ = nullptr;
    }
    
    PoolPtr& operator=(PoolPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_ && pool_) {
                pool_->destroy(ptr_);
            }
            ptr_ = other.ptr_;
            pool_ = other.pool_;
            other.ptr_ = nullptr;
            other.pool_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    PoolPtr(const PoolPtr&) = delete;
    PoolPtr& operator=(const PoolPtr&) = delete;
    
    // Access
    T* get() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    
    // Release ownership
    T* release() {
        T* tmp = ptr_;
        ptr_ = nullptr;
        pool_ = nullptr;
        return tmp;
    }
    
private:
    T* ptr_;
    ObjectPool<T>* pool_;
};

// Helper function to create pool pointer
template<typename T, typename... Args>
PoolPtr<T> makePooled(ObjectPool<T>& pool, Args&&... args) {
    return PoolPtr<T>(pool.construct(std::forward<Args>(args)...), &pool);
}

} // namespace cfd::memory