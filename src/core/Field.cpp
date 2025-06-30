
// === src/core/Field.cpp ===
#include "cfd/core/Field.hpp"
#include "cfd/core/BoundaryCondition.hpp"
#include "cfd/parallel/Communication.hpp"
#include <fstream>
#include <algorithm>

namespace cfd {

// ScalarField implementation
ScalarField::ScalarField(SharedPtr<Mesh> mesh, const std::string& name,
                        FieldLocation location)
    : FieldBase(mesh, name, FieldType::SCALAR, location) {
    
    size_t size = (location == FieldLocation::CELL) ? 
                  mesh->numCells() : mesh->numFaces();
    data_.resize(size, 0.0);
    
    if (storeOldValues_) {
        oldData_.resize(size, 0.0);
        oldOldData_.resize(size, 0.0);
    }
}

void ScalarField::initialize(const std::function<Real(const Vector3&)>& func) {
    if (location_ == FieldLocation::CELL) {
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            data_[i] = func(mesh_->cell(i).center());
        }
    } else {
        for (Index i = 0; i < mesh_->numFaces(); ++i) {
            data_[i] = func(mesh_->face(i).center());
        }
    }
    
    updateBoundaryConditions();
}

void ScalarField::updateBoundaryConditions() {
    for (auto& [patchName, bc] : boundaryConditions_) {
        bc->apply(*this);
    }
    
    // Exchange ghost cell values in parallel
    if (parallel::MPIWrapper::isParallel() && ghostUpdater_) {
        ghostUpdater_->exchange(*this);
    }
}

ScalarField& ScalarField::operator=(Real value) {
    std::fill(data_.begin(), data_.end(), value);
    updateBoundaryConditions();
    return *this;
}

ScalarField& ScalarField::operator+=(const ScalarField& rhs) {
    if (size() != rhs.size()) {
        throw std::runtime_error("Field size mismatch in operator+=");
    }
    
    for (size_t i = 0; i < size(); ++i) {
        data_[i] += rhs.data_[i];
    }
    
    updateBoundaryConditions();
    return *this;
}

ScalarField& ScalarField::operator-=(const ScalarField& rhs) {
    if (size() != rhs.size()) {
        throw std::runtime_error("Field size mismatch in operator-=");
    }
    
    for (size_t i = 0; i < size(); ++i) {
        data_[i] -= rhs.data_[i];
    }
    
    updateBoundaryConditions();
    return *this;
}

ScalarField& ScalarField::operator*=(Real scalar) {
    for (auto& value : data_) {
        value *= scalar;
    }
    
    updateBoundaryConditions();
    return *this;
}

ScalarField& ScalarField::operator/=(Real scalar) {
    Real invScalar = 1.0 / scalar;
    for (auto& value : data_) {
        value *= invScalar;
    }
    
    updateBoundaryConditions();
    return *this;
}

void ScalarField::storeOldTime() {
    if (!storeOldValues_) {
        oldData_.resize(size());
        oldOldData_.resize(size());
        storeOldValues_ = true;
    }
    
    oldOldData_ = oldData_;
    oldData_ = data_;
}

ScalarField ScalarField::oldTime() const {
    ScalarField old(mesh_, name_ + "_old", location_);
    old.data_ = oldData_;
    return old;
}

ScalarField ScalarField::oldOldTime() const {
    ScalarField oldOld(mesh_, name_ + "_oldOld", location_);
    oldOld.data_ = oldOldData_;
    return oldOld;
}

Real ScalarField::min() const {
    if (data_.empty()) return 0.0;
    
    Real localMin = *std::min_element(data_.begin(), data_.end());
    
    if (parallel::MPIWrapper::isParallel()) {
        return parallel::globalMin(localMin);
    }
    
    return localMin;
}

Real ScalarField::max() const {
    if (data_.empty()) return 0.0;
    
    Real localMax = *std::max_element(data_.begin(), data_.end());
    
    if (parallel::MPIWrapper::isParallel()) {
        return parallel::globalMax(localMax);
    }
    
    return localMax;
}

Real ScalarField::average() const {
    Real sum = 0.0;
    Real volume = 0.0;
    
    if (location_ == FieldLocation::CELL) {
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            Real vol = mesh_->cell(i).volume();
            sum += data_[i] * vol;
            volume += vol;
        }
    } else {
        for (Index i = 0; i < mesh_->numFaces(); ++i) {
            Real area = mesh_->face(i).area();
            sum += data_[i] * area;
            volume += area;
        }
    }
    
    if (parallel::MPIWrapper::isParallel()) {
        sum = parallel::globalSum(sum);
        volume = parallel::globalSum(volume);
    }
    
    return sum / (volume + SMALL);
}

void ScalarField::write(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write header
    size_t dataSize = data_.size();
    file.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));
    
    // Write data
    file.write(reinterpret_cast<const char*>(data_.data()), 
               dataSize * sizeof(Real));
}

void ScalarField::read(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Read header
    size_t dataSize;
    file.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    
    if (dataSize != data_.size()) {
        throw std::runtime_error("Field size mismatch when reading: " + filename);
    }
    
    // Read data
    file.read(reinterpret_cast<char*>(data_.data()), 
              dataSize * sizeof(Real));
    
    updateBoundaryConditions();
}

// VectorField implementation
VectorField::VectorField(SharedPtr<Mesh> mesh, const std::string& name,
                        FieldLocation location)
    : FieldBase(mesh, name, FieldType::VECTOR, location) {
    
    size_t size = (location == FieldLocation::CELL) ? 
                  mesh->numCells() : mesh->numFaces();
    data_.resize(size, Vector3::Zero());
    
    if (storeOldValues_) {
        oldData_.resize(size, Vector3::Zero());
        oldOldData_.resize(size, Vector3::Zero());
    }
}

void VectorField::initialize(const std::function<Vector3(const Vector3&)>& func) {
    if (location_ == FieldLocation::CELL) {
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            data_[i] = func(mesh_->cell(i).center());
        }
    } else {
        for (Index i = 0; i < mesh_->numFaces(); ++i) {
            data_[i] = func(mesh_->face(i).center());
        }
    }
    
    updateBoundaryConditions();
}

void VectorField::updateBoundaryConditions() {
    for (auto& [patchName, bc] : boundaryConditions_) {
        bc->apply(*this);
    }
    
    // Exchange ghost cell values in parallel
    if (parallel::MPIWrapper::isParallel() && ghostUpdater_) {
        ghostUpdater_->exchange(*this);
    }
}

VectorField& VectorField::operator=(const Vector3& value) {
    std::fill(data_.begin(), data_.end(), value);
    updateBoundaryConditions();
    return *this;
}

ScalarField VectorField::dot(const VectorField& rhs) const {
    if (size() != rhs.size()) {
        throw std::runtime_error("Field size mismatch in dot product");
    }
    
    ScalarField result(mesh_, name_ + "_dot_" + rhs.name(), location_);
    
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data_[i].dot(rhs.data_[i]);
    }
    
    return result;
}

VectorField VectorField::cross(const VectorField& rhs) const {
    if (size() != rhs.size()) {
        throw std::runtime_error("Field size mismatch in cross product");
    }
    
    VectorField result(mesh_, name_ + "_cross_" + rhs.name(), location_);
    
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data_[i].cross(rhs.data_[i]);
    }
    
    return result;
}

ScalarField VectorField::mag() const {
    ScalarField result(mesh_, name_ + "_mag", location_);
    
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data_[i].norm();
    }
    
    return result;
}

ScalarField VectorField::component(int comp) const {
    if (comp < 0 || comp > 2) {
        throw std::runtime_error("Invalid component index");
    }
    
    ScalarField result(mesh_, name_ + "_" + std::to_string(comp), location_);
    
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data_[i][comp];
    }
    
    return result;
}

void VectorField::storeOldTime() {
    if (!storeOldValues_) {
        oldData_.resize(size());
        oldOldData_.resize(size());
        storeOldValues_ = true;
    }
    
    oldOldData_ = oldData_;
    oldData_ = data_;
}

// Binary operators
ScalarField operator+(const ScalarField& lhs, const ScalarField& rhs) {
    ScalarField result = lhs;
    result += rhs;
    return result;
}

ScalarField operator-(const ScalarField& lhs, const ScalarField& rhs) {
    ScalarField result = lhs;
    result -= rhs;
    return result;
}

ScalarField operator*(const ScalarField& lhs, const ScalarField& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("Field size mismatch in operator*");
    }
    
    ScalarField result(lhs.mesh(), lhs.name() + "*" + rhs.name());
    
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    
    return result;
}

ScalarField operator/(const ScalarField& lhs, const ScalarField& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("Field size mismatch in operator/");
    }
    
    ScalarField result(lhs.mesh(), lhs.name() + "/" + rhs.name());
    
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] / (rhs[i] + SMALL);
    }
    
    return result;
}

VectorField operator+(const VectorField& lhs, const VectorField& rhs) {
    VectorField result = lhs;
    result += rhs;
    return result;
}

VectorField operator-(const VectorField& lhs, const VectorField& rhs) {
    VectorField result = lhs;
    result -= rhs;
    return result;
}

VectorField operator*(const ScalarField& lhs, const VectorField& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("Field size mismatch in operator*");
    }
    
    VectorField result(rhs.mesh(), lhs.name() + "*" + rhs.name());
    
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    
    return result;
}

VectorField operator*(const VectorField& lhs, const ScalarField& rhs) {
    return rhs * lhs;
}
