#pragma once

#include "cfd/core/Types.hpp"
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>

namespace cfd::memory {

// Aligned allocator for SIMD operations
template<typename T, std::size_t Alignment = SIMD_ALIGNMENT>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    // Rebind to different type
    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
    
    // Constructors
    AlignedAllocator() noexcept = default;
    
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    
    // Allocate aligned memory
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        
        // Check for overflow
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        
        std::size_t bytes = n * sizeof(T);
        void* p = std::aligned_alloc(Alignment, alignedSize(bytes));
        
        if (!p) {
            throw std::bad_alloc();
        }
        
        return static_cast<T*>(p);
    }
    
    // Deallocate memory
    void deallocate(T* p, std::size_t /*n*/) noexcept {
        std::free(p);
    }
    
    // Maximum allocation size
    std::size_t max_size() const noexcept {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }
    
    // Construct object
    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
    
    // Destroy object
    template<typename U>
    void destroy(U* p) noexcept {
        p->~U();
    }
    
    // Get alignment
    static constexpr std::size_t alignment() noexcept {
        return Alignment;
    }
    
private:
    // Round up to alignment boundary
    static std::size_t alignedSize(std::size_t size) noexcept {
        return (size + Alignment - 1) & ~(Alignment - 1);
    }
};

// Equality operators
template<typename T1, typename T2, std::size_t A1, std::size_t A2>
bool operator==(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept {
    return A1 == A2;
}

template<typename T1, typename T2, std::size_t A1, std::size_t A2>
bool operator!=(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept {
    return A1 != A2;
}

// Aligned deleter for unique_ptr
template<std::size_t Alignment = SIMD_ALIGNMENT>
struct AlignedDeleter {
    void operator()(void* p) const noexcept {
        std::free(p);
    }
};

// Make aligned unique_ptr
template<typename T, std::size_t Alignment = SIMD_ALIGNMENT, typename... Args>
std::unique_ptr<T, AlignedDeleter<Alignment>> makeAlignedUnique(Args&&... args) {
    void* memory = std::aligned_alloc(Alignment, sizeof(T));
    if (!memory) {
        throw std::bad_alloc();
    }
    
    try {
        T* obj = new(memory) T(std::forward<Args>(args)...);
        return std::unique_ptr<T, AlignedDeleter<Alignment>>(obj);
    } catch (...) {
        std::free(memory);
        throw;
    }
}

// Make aligned shared_ptr
template<typename T, std::size_t Alignment = SIMD_ALIGNMENT, typename... Args>
std::shared_ptr<T> makeAlignedShared(Args&&... args) {
    struct AlignedBlock {
        alignas(Alignment) std::byte storage[sizeof(T)];
    };
    
    auto block = std::make_shared<AlignedBlock>();
    T* obj = new(&block->storage) T(std::forward<Args>(args)...);
    
    return std::shared_ptr<T>(block, obj);
}

// Aligned array allocation
template<typename T, std::size_t Alignment = SIMD_ALIGNMENT>
class AlignedArray {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = std::size_t;
    
    // Constructors
    AlignedArray() noexcept : data_(nullptr), size_(0), capacity_(0) {}
    
    explicit AlignedArray(size_type n) : size_(n), capacity_(n) {
        allocate(n);
        std::uninitialized_default_construct_n(data_, n);
    }
    
    AlignedArray(size_type n, const T& value) : size_(n), capacity_(n) {
        allocate(n);
        std::uninitialized_fill_n(data_, n, value);
    }
    
    // Copy constructor
    AlignedArray(const AlignedArray& other) : size_(other.size_), capacity_(other.size_) {
        if (size_ > 0) {
            allocate(size_);
            std::uninitialized_copy_n(other.data_, size_, data_);
        }
    }
    
    // Move constructor
    AlignedArray(AlignedArray&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    // Destructor
    ~AlignedArray() {
        clear();
        deallocate();
    }
    
    // Assignment operators
    AlignedArray& operator=(const AlignedArray& other) {
        if (this != &other) {
            AlignedArray tmp(other);
            swap(tmp);
        }
        return *this;
    }
    
    AlignedArray& operator=(AlignedArray&& other) noexcept {
        if (this != &other) {
            clear();
            deallocate();
            
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    // Element access
    reference operator[](size_type i) { return data_[i]; }
    const_reference operator[](size_type i) const { return data_[i]; }
    
    reference at(size_type i) {
        if (i >= size_) throw std::out_of_range("AlignedArray: index out of range");
        return data_[i];
    }
    
    const_reference at(size_type i) const {
        if (i >= size_) throw std::out_of_range("AlignedArray: index out of range");
        return data_[i];
    }
    
    pointer data() noexcept { return data_; }
    const_pointer data() const noexcept { return data_; }
    
    // Iterators
    iterator begin() noexcept { return data_; }
    const_iterator begin() const noexcept { return data_; }
    iterator end() noexcept { return data_ + size_; }
    const_iterator end() const noexcept { return data_ + size_; }
    
    // Capacity
    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    
    void reserve(size_type new_cap) {
        if (new_cap > capacity_) {
            reallocate(new_cap);
        }
    }
    
    void resize(size_type new_size) {
        if (new_size > size_) {
            if (new_size > capacity_) {
                reallocate(new_size);
            }
            std::uninitialized_default_construct_n(data_ + size_, new_size - size_);
        } else if (new_size < size_) {
            std::destroy_n(data_ + new_size, size_ - new_size);
        }
        size_ = new_size;
    }
    
    void resize(size_type new_size, const T& value) {
        if (new_size > size_) {
            if (new_size > capacity_) {
                reallocate(new_size);
            }
            std::uninitialized_fill_n(data_ + size_, new_size - size_, value);
        } else if (new_size < size_) {
            std::destroy_n(data_ + new_size, size_ - new_size);
        }
        size_ = new_size;
    }
    
    // Modifiers
    void push_back(const T& value) {
        if (size_ == capacity_) {
            reallocate(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new(data_ + size_) T(value);
        ++size_;
    }
    
    void push_back(T&& value) {
        if (size_ == capacity_) {
            reallocate(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new(data_ + size_) T(std::move(value));
        ++size_;
    }
    
    template<typename... Args>
    void emplace_back(Args&&... args) {
        if (size_ == capacity_) {
            reallocate(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new(data_ + size_) T(std::forward<Args>(args)...);
        ++size_;
    }
    
    void pop_back() {
        if (size_ > 0) {
            --size_;
            data_[size_].~T();
        }
    }
    
    void clear() noexcept {
        std::destroy_n(data_, size_);
        size_ = 0;
    }
    
    void swap(AlignedArray& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }
    
private:
    T* data_;
    size_type size_;
    size_type capacity_;
    
    void allocate(size_type n) {
        if (n > 0) {
            void* memory = std::aligned_alloc(Alignment, n * sizeof(T));
            if (!memory) {
                throw std::bad_alloc();
            }
            data_ = static_cast<T*>(memory);
        }
    }
    
    void deallocate() noexcept {
        std::free(data_);
        data_ = nullptr;
        capacity_ = 0;
    }
    
    void reallocate(size_type new_cap) {
        T* new_data = nullptr;
        
        if (new_cap > 0) {
            void* memory = std::aligned_alloc(Alignment, new_cap * sizeof(T));
            if (!memory) {
                throw std::bad_alloc();
            }
            new_data = static_cast<T*>(memory);
            
            // Move existing elements
            if (data_) {
                std::uninitialized_move_n(data_, size_, new_data);
                std::destroy_n(data_, size_);
            }
        }
        
        deallocate();
        data_ = new_data;
        capacity_ = new_cap;
    }
};

// Type aliases for common alignments
template<typename T>
using CacheAlignedAllocator = AlignedAllocator<T, CACHE_LINE_SIZE>;

template<typename T>
using SimdAlignedAllocator = AlignedAllocator<T, SIMD_ALIGNMENT>;

// Check alignment
template<typename T>
bool isAligned(const T* ptr, std::size_t alignment) noexcept {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

// Alignment traits
template<typename T>
struct AlignmentTraits {
    static constexpr std::size_t value = alignof(T);
    static constexpr bool is_simd_aligned = value >= SIMD_ALIGNMENT;
    static constexpr bool is_cache_aligned = value >= CACHE_LINE_SIZE;
};

} // namespace cfd::memory