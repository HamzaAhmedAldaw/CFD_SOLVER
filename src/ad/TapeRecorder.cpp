// === src/ad/TapeRecorder.cpp ===
#include "cfd/ad/TapeRecorder.hpp"
#include <algorithm>
#include <stdexcept>

namespace cfd::ad {

// Variable assignment operators implementation
TapeRecorder::Variable& TapeRecorder::Variable::operator+=(const Variable& rhs) {
    *this = *this + rhs;
    return *this;
}

TapeRecorder::Variable& TapeRecorder::Variable::operator-=(const Variable& rhs) {
    *this = *this - rhs;
    return *this;
}

TapeRecorder::Variable& TapeRecorder::Variable::operator*=(const Variable& rhs) {
    *this = *this * rhs;
    return *this;
}

TapeRecorder::Variable& TapeRecorder::Variable::operator/=(const Variable& rhs) {
    *this = *this / rhs;
    return *this;
}

// Power function with variable exponent
TapeRecorder::Variable TapeRecorder::pow(const Variable& x, const Variable& p) {
    Real xval = x.value();
    Real pval = p.value();
    Real val = std::pow(xval, pval);
    
    // Derivatives: d/dx(x^p) = p*x^(p-1), d/dp(x^p) = x^p*ln(x)
    Real dfdx = pval * std::pow(xval, pval - 1.0);
    Real dfdp = val * std::log(xval);
    
    return recordOp(OpType::POW, {x.index(), p.index()}, val, {dfdx, dfdp});
}

// Memory pool for tape operations to reduce allocations
class TapeMemoryPool {
public:
    static TapeMemoryPool& instance() {
        static TapeMemoryPool pool;
        return pool;
    }
    
    void* allocate(size_t size) {
        if (size <= sizeof(TapeRecorder::Operation)) {
            if (freeList_.empty()) {
                expandPool();
            }
            void* ptr = freeList_.back();
            freeList_.pop_back();
            return ptr;
        }
        return ::operator new(size);
    }
    
    void deallocate(void* ptr, size_t size) {
        if (size <= sizeof(TapeRecorder::Operation)) {
            freeList_.push_back(ptr);
        } else {
            ::operator delete(ptr);
        }
    }
    
private:
    std::vector<void*> freeList_;
    std::vector<std::unique_ptr<char[]>> chunks_;
    
    void expandPool() {
        const size_t chunkSize = 1024 * sizeof(TapeRecorder::Operation);
        auto chunk = std::make_unique<char[]>(chunkSize);
        
        for (size_t i = 0; i < chunkSize; i += sizeof(TapeRecorder::Operation)) {
            freeList_.push_back(&chunk[i]);
        }
        
        chunks_.push_back(std::move(chunk));
    }
};

} // namespace cfd::ad
