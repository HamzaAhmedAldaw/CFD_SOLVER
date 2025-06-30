#pragma once

#include "cfd/core/Types.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <stack>

namespace cfd::ad {

// Tape recorder for reverse-mode automatic differentiation
class TapeRecorder {
public:
    // Forward declaration
    class Variable;
    
    // Operation types
    enum class OpType {
        CONST,
        INPUT,
        ADD,
        SUB,
        MUL,
        DIV,
        NEG,
        SQRT,
        EXP,
        LOG,
        SIN,
        COS,
        TAN,
        POW,
        ABS,
        MIN,
        MAX
    };
    
    // Operation record
    struct Operation {
        OpType type;
        std::vector<size_t> inputs;
        size_t output;
        Real value;
        std::vector<Real> partials;  // Partial derivatives
        
        Operation(OpType t, size_t out) 
            : type(t), output(out), value(0.0) {}
    };
    
    // Variable handle for tape operations
    class Variable {
    public:
        Variable() : tape_(nullptr), index_(0) {}
        Variable(TapeRecorder* tape, size_t index) 
            : tape_(tape), index_(index) {}
        
        // Access
        Real value() const { 
            return tape_ ? tape_->values_[index_] : 0.0; 
        }
        
        size_t index() const { return index_; }
        bool valid() const { return tape_ != nullptr; }
        
        // Arithmetic operations
        Variable operator+(const Variable& rhs) const;
        Variable operator-(const Variable& rhs) const;
        Variable operator*(const Variable& rhs) const;
        Variable operator/(const Variable& rhs) const;
        Variable operator-() const;
        
        Variable operator+(Real rhs) const;
        Variable operator-(Real rhs) const;
        Variable operator*(Real rhs) const;
        Variable operator/(Real rhs) const;
        
        // Assignment
        Variable& operator+=(const Variable& rhs);
        Variable& operator-=(const Variable& rhs);
        Variable& operator*=(const Variable& rhs);
        Variable& operator/=(const Variable& rhs);
        
    private:
        TapeRecorder* tape_;
        size_t index_;
        
        friend class TapeRecorder;
    };
    
    TapeRecorder() : recording_(true) {}
    
    // Create new variable
    Variable newVariable(Real value) {
        size_t index = values_.size();
        values_.push_back(value);
        adjoints_.push_back(0.0);
        
        if (recording_) {
            operations_.emplace_back(OpType::INPUT, index);
        }
        
        return Variable(this, index);
    }
    
    // Create constant (not differentiated)
    Variable constant(Real value) {
        size_t index = values_.size();
        values_.push_back(value);
        adjoints_.push_back(0.0);
        
        if (recording_) {
            operations_.emplace_back(OpType::CONST, index);
        }
        
        return Variable(this, index);
    }
    
    // Compute gradient via backpropagation
    void gradient(const Variable& output, const std::vector<Variable>& inputs) {
        // Clear adjoints
        std::fill(adjoints_.begin(), adjoints_.end(), 0.0);
        
        // Seed output
        adjoints_[output.index()] = 1.0;
        
        // Backward pass
        for (auto it = operations_.rbegin(); it != operations_.rend(); ++it) {
            backpropagate(*it);
        }
    }
    
    // Get derivative
    Real derivative(const Variable& output, const Variable& input) const {
        return adjoints_[input.index()];
    }
    
    // Control recording
    void startRecording() { recording_ = true; }
    void stopRecording() { recording_ = false; }
    bool isRecording() const { return recording_; }
    
    // Clear tape
    void reset() {
        values_.clear();
        adjoints_.clear();
        operations_.clear();
    }
    
    // Mathematical functions
    Variable sqrt(const Variable& x);
    Variable exp(const Variable& x);
    Variable log(const Variable& x);
    Variable sin(const Variable& x);
    Variable cos(const Variable& x);
    Variable tan(const Variable& x);
    Variable pow(const Variable& x, Real p);
    Variable pow(const Variable& x, const Variable& p);
    Variable abs(const Variable& x);
    Variable min(const Variable& a, const Variable& b);
    Variable max(const Variable& a, const Variable& b);
    
private:
    std::vector<Real> values_;      // Variable values
    std::vector<Real> adjoints_;    // Adjoint values (derivatives)
    std::vector<Operation> operations_;  // Operation tape
    bool recording_;
    
    // Record operation
    Variable recordOp(OpType type, const std::vector<size_t>& inputs,
                     Real value, const std::vector<Real>& partials) {
        size_t index = values_.size();
        values_.push_back(value);
        adjoints_.push_back(0.0);
        
        if (recording_) {
            Operation& op = operations_.emplace_back(type, index);
            op.inputs = inputs;
            op.value = value;
            op.partials = partials;
        }
        
        return Variable(this, index);
    }
    
    // Backpropagation for each operation type
    void backpropagate(const Operation& op);
    
    // Friend functions for external operators
    friend Variable operator+(Real lhs, const Variable& rhs);
    friend Variable operator-(Real lhs, const Variable& rhs);
    friend Variable operator*(Real lhs, const Variable& rhs);
    friend Variable operator/(Real lhs, const Variable& rhs);
};

// Implementation of Variable operations
inline TapeRecorder::Variable TapeRecorder::Variable::operator+(const Variable& rhs) const {
    if (!tape_ || tape_ != rhs.tape_) {
        throw std::runtime_error("Invalid tape operation");
    }
    
    Real val = value() + rhs.value();
    return tape_->recordOp(OpType::ADD, {index_, rhs.index_}, val, {1.0, 1.0});
}

inline TapeRecorder::Variable TapeRecorder::Variable::operator-(const Variable& rhs) const {
    if (!tape_ || tape_ != rhs.tape_) {
        throw std::runtime_error("Invalid tape operation");
    }
    
    Real val = value() - rhs.value();
    return tape_->recordOp(OpType::SUB, {index_, rhs.index_}, val, {1.0, -1.0});
}

inline TapeRecorder::Variable TapeRecorder::Variable::operator*(const Variable& rhs) const {
    if (!tape_ || tape_ != rhs.tape_) {
        throw std::runtime_error("Invalid tape operation");
    }
    
    Real val = value() * rhs.value();
    return tape_->recordOp(OpType::MUL, {index_, rhs.index_}, val, 
                          {rhs.value(), value()});
}

inline TapeRecorder::Variable TapeRecorder::Variable::operator/(const Variable& rhs) const {
    if (!tape_ || tape_ != rhs.tape_) {
        throw std::runtime_error("Invalid tape operation");
    }
    
    Real val = value() / rhs.value();
    Real rhs_val2 = rhs.value() * rhs.value();
    return tape_->recordOp(OpType::DIV, {index_, rhs.index_}, val,
                          {1.0/rhs.value(), -value()/rhs_val2});
}

inline TapeRecorder::Variable TapeRecorder::Variable::operator-() const {
    if (!tape_) {
        throw std::runtime_error("Invalid tape operation");
    }
    
    Real val = -value();
    return tape_->recordOp(OpType::NEG, {index_}, val, {-1.0});
}

// Scalar operations
inline TapeRecorder::Variable TapeRecorder::Variable::operator+(Real rhs) const {
    return *this + tape_->constant(rhs);
}

inline TapeRecorder::Variable TapeRecorder::Variable::operator-(Real rhs) const {
    return *this - tape_->constant(rhs);
}

inline TapeRecorder::Variable TapeRecorder::Variable::operator*(Real rhs) const {
    return *this * tape_->constant(rhs);
}

inline TapeRecorder::Variable TapeRecorder::Variable::operator/(Real rhs) const {
    return *this / tape_->constant(rhs);
}

// Mathematical functions implementation
inline TapeRecorder::Variable TapeRecorder::sqrt(const Variable& x) {
    Real val = std::sqrt(x.value());
    Real deriv = 0.5 / val;
    return recordOp(OpType::SQRT, {x.index()}, val, {deriv});
}

inline TapeRecorder::Variable TapeRecorder::exp(const Variable& x) {
    Real val = std::exp(x.value());
    return recordOp(OpType::EXP, {x.index()}, val, {val});
}

inline TapeRecorder::Variable TapeRecorder::log(const Variable& x) {
    Real val = std::log(x.value());
    Real deriv = 1.0 / x.value();
    return recordOp(OpType::LOG, {x.index()}, val, {deriv});
}

inline TapeRecorder::Variable TapeRecorder::sin(const Variable& x) {
    Real val = std::sin(x.value());
    Real deriv = std::cos(x.value());
    return recordOp(OpType::SIN, {x.index()}, val, {deriv});
}

inline TapeRecorder::Variable TapeRecorder::cos(const Variable& x) {
    Real val = std::cos(x.value());
    Real deriv = -std::sin(x.value());
    return recordOp(OpType::COS, {x.index()}, val, {deriv});
}

inline TapeRecorder::Variable TapeRecorder::tan(const Variable& x) {
    Real val = std::tan(x.value());
    Real sec2 = 1.0 + val * val;
    return recordOp(OpType::TAN, {x.index()}, val, {sec2});
}

inline TapeRecorder::Variable TapeRecorder::pow(const Variable& x, Real p) {
    Real val = std::pow(x.value(), p);
    Real deriv = p * std::pow(x.value(), p - 1.0);
    return recordOp(OpType::POW, {x.index()}, val, {deriv});
}

inline TapeRecorder::Variable TapeRecorder::abs(const Variable& x) {
    Real val = std::abs(x.value());
    Real deriv = x.value() >= 0 ? 1.0 : -1.0;
    return recordOp(OpType::ABS, {x.index()}, val, {deriv});
}

inline TapeRecorder::Variable TapeRecorder::min(const Variable& a, const Variable& b) {
    if (a.value() <= b.value()) {
        return recordOp(OpType::MIN, {a.index(), b.index()}, a.value(), {1.0, 0.0});
    } else {
        return recordOp(OpType::MIN, {a.index(), b.index()}, b.value(), {0.0, 1.0});
    }
}

inline TapeRecorder::Variable TapeRecorder::max(const Variable& a, const Variable& b) {
    if (a.value() >= b.value()) {
        return recordOp(OpType::MAX, {a.index(), b.index()}, a.value(), {1.0, 0.0});
    } else {
        return recordOp(OpType::MAX, {a.index(), b.index()}, b.value(), {0.0, 1.0});
    }
}

// Backpropagation implementation
inline void TapeRecorder::backpropagate(const Operation& op) {
    Real adjoint = adjoints_[op.output];
    
    if (adjoint == 0.0) return;  // No contribution
    
    switch (op.type) {
        case OpType::ADD:
        case OpType::SUB:
        case OpType::MUL:
        case OpType::DIV:
            for (size_t i = 0; i < op.inputs.size(); ++i) {
                adjoints_[op.inputs[i]] += adjoint * op.partials[i];
            }
            break;
            
        case OpType::NEG:
        case OpType::SQRT:
        case OpType::EXP:
        case OpType::LOG:
        case OpType::SIN:
        case OpType::COS:
        case OpType::TAN:
        case OpType::POW:
        case OpType::ABS:
            adjoints_[op.inputs[0]] += adjoint * op.partials[0];
            break;
            
        case OpType::MIN:
        case OpType::MAX:
            for (size_t i = 0; i < op.inputs.size(); ++i) {
                adjoints_[op.inputs[i]] += adjoint * op.partials[i];
            }
            break;
            
        case OpType::CONST:
        case OpType::INPUT:
            // No backpropagation needed
            break;
    }
}

// External operators
inline TapeRecorder::Variable operator+(Real lhs, const TapeRecorder::Variable& rhs) {
    return rhs.tape_->constant(lhs) + rhs;
}

inline TapeRecorder::Variable operator-(Real lhs, const TapeRecorder::Variable& rhs) {
    return rhs.tape_->constant(lhs) - rhs;
}

inline TapeRecorder::Variable operator*(Real lhs, const TapeRecorder::Variable& rhs) {
    return rhs.tape_->constant(lhs) * rhs;
}

inline TapeRecorder::Variable operator/(Real lhs, const TapeRecorder::Variable& rhs) {
    return rhs.tape_->constant(lhs) / rhs;
}

// Checkpointing for memory efficiency
class CheckpointingTape : public TapeRecorder {
public:
    // Set checkpoint interval
    void setCheckpointInterval(size_t interval) {
        checkpointInterval_ = interval;
    }
    
    // Create checkpoint
    void checkpoint() {
        checkpoints_.push_back({values_, operations_.size()});
    }
    
    // Restore from checkpoint
    void restoreCheckpoint(size_t checkpoint) {
        if (checkpoint < checkpoints_.size()) {
            values_ = checkpoints_[checkpoint].values;
            operations_.resize(checkpoints_[checkpoint].opIndex);
        }
    }
    
private:
    struct Checkpoint {
        std::vector<Real> values;
        size_t opIndex;
    };
    
    std::vector<Checkpoint> checkpoints_;
    size_t checkpointInterval_ = 1000;
};

} // namespace cfd::ad