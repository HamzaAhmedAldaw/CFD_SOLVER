#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Mesh.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/CFDSolver.hpp"
#include <string>
#include <map>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <json/json.h>

namespace cfd::io {

// Case file reader for different formats
class CaseReader {
public:
    CaseReader(const std::string& caseDirectory)
        : caseDir_(caseDirectory) {}
    
    // Read case configuration
    void readCase(CFDSolver::SimulationSettings& settings,
                  std::map<std::string, std::any>& customSettings);
    
    // Read mesh
    SharedPtr<Mesh> readMesh();
    
    // Read initial conditions
    void readInitialConditions(std::map<std::string, SharedPtr<FieldBase>>& fields);
    
    // Read boundary conditions
    void readBoundaryConditions(std::map<std::string, SharedPtr<FieldBase>>& fields);
    
    // Read physical properties
    void readPhysicalProperties(std::map<std::string, Real>& properties);
    
    // Read numerical schemes
    void readNumericalSchemes(std::map<std::string, std::string>& schemes);
    
    // Read solver settings
    void readSolverSettings(std::map<std::string, std::any>& settings);
    
private:
    std::string caseDir_;
    
    // File format detection
    enum class Format { OPENFOAM, YAML, JSON, XML };
    Format detectFormat() const;
    
    // Format-specific readers
    void readOpenFOAMCase(CFDSolver::SimulationSettings& settings);
    void readYAMLCase(CFDSolver::SimulationSettings& settings);
    void readJSONCase(CFDSolver::SimulationSettings& settings);
    
    // OpenFOAM dictionary parser
    class OpenFOAMDict {
    public:
        OpenFOAMDict(const std::string& filename);
        
        template<typename T>
        T lookup(const std::string& keyword) const;
        
        template<typename T>
        T lookupOrDefault(const std::string& keyword, const T& defaultValue) const;
        
        bool found(const std::string& keyword) const;
        
        OpenFOAMDict subDict(const std::string& name) const;
        
    private:
        std::map<std::string, std::any> entries_;
        
        void parse(std::istream& is);
        std::any parseValue(const std::string& value);
    };
    
    // Utility functions
    std::string resolvePath(const std::string& filename) const;
    bool fileExists(const std::string& filename) const;
    
    // Time directory management
    std::vector<Real> getTimeDirs() const;
    std::string getLatestTimeDir() const;
};

// Field base class for polymorphic storage
class FieldBase {
public:
    virtual ~FieldBase() = default;
    virtual FieldType type() const = 0;
    virtual const std::string& name() const = 0;
};

// Typed field wrapper
template<typename T>
class TypedField : public FieldBase {
public:
    TypedField(SharedPtr<Field<T>> field) : field_(field) {}
    
    FieldType type() const override { return field_->fieldType(); }
    const std::string& name() const override { return field_->name(); }
    
    SharedPtr<Field<T>> field() { return field_; }
    
private:
    SharedPtr<Field<T>> field_;
};

// Configuration file writer
class CaseWriter {
public:
    CaseWriter(const std::string& caseDirectory)
        : caseDir_(caseDirectory) {}
    
    // Write case files
    void writeCase(const CFDSolver::SimulationSettings& settings);
    
    // Write mesh
    void writeMesh(const Mesh& mesh);
    
    // Write fields
    void writeFields(const std::map<std::string, SharedPtr<FieldBase>>& fields,
                    Real time);
    
    // Write convergence history
    void writeConvergenceHistory(const std::vector<Real>& residuals,
                                const std::string& fieldName);
    
    // Write probe data
    void writeProbeData(const std::vector<Vector3>& probeLocations,
                       const std::map<std::string, std::vector<Real>>& probeData,
                       Real time);
    
private:
    std::string caseDir_;
    
    // Create directory structure
    void createDirectories();
    
    // Format-specific writers
    void writeOpenFOAMDict(const std::string& filename,
                          const std::map<std::string, std::any>& dict);
    
    void writeYAMLFile(const std::string& filename,
                      const YAML::Node& node);
    
    void writeJSONFile(const std::string& filename,
                      const Json::Value& root);
};

// Parameter study reader
class ParameterStudy {
public:
    struct Parameter {
        std::string name;
        std::string path;  // Path in configuration (e.g., "solver/relaxation/U")
        std::vector<Real> values;
    };
    
    struct Case {
        std::string name;
        std::map<std::string, Real> parameters;
        std::string directory;
    };
    
    ParameterStudy(const std::string& studyFile);
    
    // Get all cases
    const std::vector<Case>& cases() const { return cases_; }
    
    // Generate cases from parameter ranges
    void generateCases();
    
    // Write case files
    void writeCases(const std::string& baseDir,
                   const CFDSolver::SimulationSettings& baseSettings);
    
private:
    std::vector<Parameter> parameters_;
    std::vector<Case> cases_;
    
    // Sampling methods
    enum class SamplingMethod { FULL_FACTORIAL, LATIN_HYPERCUBE, RANDOM };
    SamplingMethod method_ = SamplingMethod::FULL_FACTORIAL;
    
    void fullFactorial();
    void latinHypercube(int numSamples);
    void randomSampling(int numSamples);
};

// Restart file handler
class RestartIO {
public:
    RestartIO(const std::string& caseDir) : caseDir_(caseDir) {}
    
    // Write restart data
    void writeRestart(const std::map<std::string, SharedPtr<FieldBase>>& fields,
                     Real time,
                     const std::map<std::string, std::any>& metadata = {});
    
    // Read restart data
    void readRestart(std::map<std::string, SharedPtr<FieldBase>>& fields,
                    Real& time,
                    std::map<std::string, std::any>& metadata);
    
    // List available restart times
    std::vector<Real> getRestartTimes() const;
    
    // Clean old restart files
    void cleanOldRestarts(int keepLast = 5);
    
private:
    std::string caseDir_;
    
    // Binary format for efficiency
    void writeBinaryField(const std::string& filename,
                         const FieldBase& field);
    
    void readBinaryField(const std::string& filename,
                        FieldBase& field);
};

// Configuration validator
class ConfigValidator {
public:
    struct ValidationResult {
        bool valid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };
    
    // Validate configuration
    static ValidationResult validate(const CFDSolver::SimulationSettings& settings,
                                   const Mesh& mesh);
    
    // Check numerical stability
    static ValidationResult checkNumericalStability(
        const CFDSolver::SimulationSettings& settings,
        const std::map<std::string, Real>& properties);
    
    // Check boundary condition consistency
    static ValidationResult checkBoundaryConditions(
        const std::map<std::string, SharedPtr<FieldBase>>& fields,
        const Mesh& mesh);
    
private:
    // Check CFL condition
    static bool checkCFL(Real dt, Real cellSize, Real velocity);
    
    // Check diffusion number
    static bool checkDiffusionNumber(Real dt, Real cellSize, Real diffusivity);
    
    // Check mesh quality requirements
    static bool checkMeshQuality(const Mesh& mesh, Real minQuality = 0.1);
};