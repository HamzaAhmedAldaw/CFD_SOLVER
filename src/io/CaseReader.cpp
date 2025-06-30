// ===== I/O AND MEMORY MANAGEMENT IMPLEMENTATIONS =====

// === src/io/CaseReader.cpp ===
#include "cfd/io/CaseReader.hpp"
#include "cfd/core/BoundaryCondition.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>

namespace cfd::io {

namespace fs = std::filesystem;

CaseReader::CaseReader(const std::string& caseDirectory)
    : caseDir_(caseDirectory) {
    
    if (!fs::exists(caseDir_)) {
        throw std::runtime_error("Case directory does not exist: " + caseDir_);
    }
    
    // Load main configuration file
    loadConfiguration();
}

SharedPtr<Mesh> CaseReader::readMesh() {
    logger_->info("Reading mesh");
    
    YAML::Node meshConfig = config_["mesh"];
    
    if (!meshConfig) {
        throw std::runtime_error("No mesh configuration found");
    }
    
    std::string meshType = meshConfig["type"].as<std::string>("file");
    
    if (meshType == "structured") {
        return readStructuredMesh(meshConfig);
    } else if (meshType == "file") {
        std::string meshFile = meshConfig["file"].as<std::string>();
        return readMeshFile(caseDir_ + "/" + meshFile);
    } else {
        throw std::runtime_error("Unknown mesh type: " + meshType);
    }
}

SharedPtr<Mesh> CaseReader::readStructuredMesh(const YAML::Node& config) {
    auto mesh = std::make_shared<Mesh>();
    
    // Read dimensions
    std::vector<int> dims = config["dimensions"].as<std::vector<int>>();
    if (dims.size() != 3) {
        throw std::runtime_error("Structured mesh requires 3 dimensions");
    }
    
    int nx = dims[0], ny = dims[1], nz = dims[2];
    
    // Read bounds
    YAML::Node bounds = config["bounds"];
    Vector3 min = parseVector3(bounds["min"]);
    Vector3 max = parseVector3(bounds["max"]);
    
    // Generate vertices
    std::vector<Vector3> vertices;
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                Real x = min.x() + (max.x() - min.x()) * i / nx;
                Real y = min.y() + (max.y() - min.y()) * j / ny;
                Real z = min.z() + (max.z() - min.z()) * k / nz;
                vertices.emplace_back(x, y, z);
            }
        }
    }
    
    // Generate hexahedral cells
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                std::vector<Index> cellVerts(8);
                cellVerts[0] = i + j*(nx+1) + k*(nx+1)*(ny+1);
                cellVerts[1] = (i+1) + j*(nx+1) + k*(nx+1)*(ny+1);
                cellVerts[2] = (i+1) + (j+1)*(nx+1) + k*(nx+1)*(ny+1);
                cellVerts[3] = i + (j+1)*(nx+1) + k*(nx+1)*(ny+1);
                cellVerts[4] = i + j*(nx+1) + (k+1)*(nx+1)*(ny+1);
                cellVerts[5] = (i+1) + j*(nx+1) + (k+1)*(nx+1)*(ny+1);
                cellVerts[6] = (i+1) + (j+1)*(nx+1) + (k+1)*(nx+1)*(ny+1);
                cellVerts[7] = i + (j+1)*(nx+1) + (k+1)*(nx+1)*(ny+1);
                
                mesh->addCell(CellType::HEXAHEDRON, cellVerts);
            }
        }
    }
    
    // Build mesh connectivity
    mesh->build(vertices);
    
    // Add boundary patches
    mesh->addBoundaryPatch("left", BCType::WALL);
    mesh->addBoundaryPatch("right", BCType::WALL);
    mesh->addBoundaryPatch("bottom", BCType::WALL);
    mesh->addBoundaryPatch("top", BCType::WALL);
    mesh->addBoundaryPatch("front", BCType::WALL);
    mesh->addBoundaryPatch("back", BCType::WALL);
    
    // Assign faces to patches based on position
    assignBoundaryPatches(mesh, min, max);
    
    logger_->info("Created structured mesh: {} cells", mesh->numCells());
    
    return mesh;
}

SharedPtr<Mesh> CaseReader::readMeshFile(const std::string& filename) {
    auto mesh = std::make_shared<Mesh>();
    
    // Determine file format
    std::string ext = fs::path(filename).extension();
    
    if (ext == ".msh") {
        readGmshFile(filename, mesh);
    } else if (ext == ".cas") {
        readFluentFile(filename, mesh);
    } else if (ext == ".cgns") {
        readCGNSFile(filename, mesh);
    } else {
        throw std::runtime_error("Unsupported mesh format: " + ext);
    }
    
    return mesh;
}

void CaseReader::readGmshFile(const std::string& filename, SharedPtr<Mesh> mesh) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open mesh file: " + filename);
    }
    
    std::string line;
    
    // Read format
    std::getline(file, line); // $MeshFormat
    std::getline(file, line); // version file-type data-size
    std::getline(file, line); // $EndMeshFormat
    
    // Read nodes
    std::getline(file, line); // $Nodes
    int numNodes;
    file >> numNodes;
    
    std::vector<Vector3> vertices(numNodes);
    std::map<int, Index> nodeMap;
    
    for (int i = 0; i < numNodes; ++i) {
        int nodeId;
        Real x, y, z;
        file >> nodeId >> x >> y >> z;
        vertices[i] = Vector3(x, y, z);
        nodeMap[nodeId] = i;
    }
    
    std::getline(file, line); // consume newline
    std::getline(file, line); // $EndNodes
    
    // Read elements
    std::getline(file, line); // $Elements
    int numElements;
    file >> numElements;
    
    for (int i = 0; i < numElements; ++i) {
        int elemId, elemType, numTags;
        file >> elemId >> elemType >> numTags;
        
        // Skip tags
        for (int j = 0; j < numTags; ++j) {
            int tag;
            file >> tag;
        }
        
        // Read nodes based on element type
        std::vector<Index> cellVerts;
        
        switch (elemType) {
            case 4: // Tetrahedron
                cellVerts.resize(4);
                for (int j = 0; j < 4; ++j) {
                    int nodeId;
                    file >> nodeId;
                    cellVerts[j] = nodeMap[nodeId];
                }
                mesh->addCell(CellType::TETRAHEDRON, cellVerts);
                break;
                
            case 5: // Hexahedron
                cellVerts.resize(8);
                for (int j = 0; j < 8; ++j) {
                    int nodeId;
                    file >> nodeId;
                    cellVerts[j] = nodeMap[nodeId];
                }
                mesh->addCell(CellType::HEXAHEDRON, cellVerts);
                break;
                
            // Add other element types as needed
        }
    }
    
    // Build mesh
    mesh->build(vertices);
}

void CaseReader::assignBoundaryPatches(SharedPtr<Mesh> mesh,
                                      const Vector3& min,
                                      const Vector3& max) {
    const Real tol = 1e-6;
    
    for (Index faceId = 0; faceId < mesh->numFaces(); ++faceId) {
        Face& face = mesh->face(faceId);
        
        if (face.isBoundary()) {
            const Vector3& center = face.center();
            
            if (std::abs(center.x() - min.x()) < tol) {
                face.setPatchId(mesh->boundaryPatch("left").id());
            } else if (std::abs(center.x() - max.x()) < tol) {
                face.setPatchId(mesh->boundaryPatch("right").id());
            } else if (std::abs(center.y() - min.y()) < tol) {
                face.setPatchId(mesh->boundaryPatch("bottom").id());
            } else if (std::abs(center.y() - max.y()) < tol) {
                face.setPatchId(mesh->boundaryPatch("top").id());
            } else if (std::abs(center.z() - min.z()) < tol) {
                face.setPatchId(mesh->boundaryPatch("front").id());
            } else if (std::abs(center.z() - max.z()) < tol) {
                face.setPatchId(mesh->boundaryPatch("back").id());
            }
        }
    }
}

CFDSolver::SimulationSettings CaseReader::readSimulationSettings() {
    CFDSolver::SimulationSettings settings;
    
    YAML::Node simNode = config_["simulation"];
    if (!simNode) {
        throw std::runtime_error("No simulation settings found");
    }
    
    // Solver type
    std::string solverType = simNode["type"].as<std::string>("incompressible");
    if (solverType == "incompressible") {
        settings.solverType = SolverType::INCOMPRESSIBLE;
    } else if (solverType == "compressible") {
        settings.solverType = SolverType::COMPRESSIBLE;
    } else if (solverType == "lowMach") {
        settings.solverType = SolverType::LOW_MACH;
    }
    
    // Time settings
    settings.transient = simNode["transient"].as<bool>(false);
    settings.startTime = simNode["startTime"].as<Real>(0.0);
    settings.endTime = simNode["endTime"].as<Real>(1.0);
    settings.deltaT = simNode["deltaT"].as<Real>(0.001);
    settings.adjustTimeStep = simNode["adjustTimeStep"].as<bool>(false);
    settings.maxCo = simNode["maxCo"].as<Real>(0.5);
    settings.maxDeltaT = simNode["maxDeltaT"].as<Real>(1.0);
    
    // Output settings
    settings.writeInterval = simNode["writeInterval"].as<Real>(0.1);
    settings.writeFrequency = simNode["writeFrequency"].as<int>(0);
    
    // Turbulence
    YAML::Node turbNode = config_["turbulence"];
    if (turbNode) {
        std::string model = turbNode["model"].as<std::string>("laminar");
        if (model != "laminar") {
            settings.turbulence = true;
            
            if (model == "SpalartAllmaras") {
                settings.turbulenceModel = TurbulenceModelType::SPALART_ALLMARAS;
            } else if (model == "kEpsilon") {
                settings.turbulenceModel = TurbulenceModelType::K_EPSILON;
            } else if (model == "kOmega") {
                settings.turbulenceModel = TurbulenceModelType::K_OMEGA;
            } else if (model == "kOmegaSST") {
                settings.turbulenceModel = TurbulenceModelType::K_OMEGA_SST;
            }
        }
    }
    
    // Convergence
    YAML::Node convNode = config_["convergence"];
    if (convNode) {
        settings.maxIterations = convNode["nIterations"].as<int>(1000);
        settings.convergenceTolerance = convNode["tolerance"].as<Real>(1e-6);
    }
    
    return settings;
}

physics::PhysicsParameters CaseReader::readPhysicsParameters() {
    physics::PhysicsParameters params;
    
    YAML::Node physicsNode = config_["physics"];
    if (!physicsNode) {
        throw std::runtime_error("No physics parameters found");
    }
    
    // Basic properties
    params.rho0 = physicsNode["density"].as<Real>(1.0);
    params.mu0 = physicsNode["viscosity"].as<Real>(1e-3);
    params.T0 = physicsNode["temperature"].as<Real>(293.15);
    params.p0 = physicsNode["pressure"].as<Real>(101325.0);
    
    // Gravity
    if (physicsNode["gravity"]) {
        params.gravity = parseVector3(physicsNode["gravity"]);
    }
    
    // Gas properties
    params.R = physicsNode["gasConstant"].as<Real>(287.0);
    params.gamma = physicsNode["gamma"].as<Real>(1.4);
    params.Pr = physicsNode["Pr"].as<Real>(0.72);
    params.Prt = physicsNode["Prt"].as<Real>(0.85);
    
    // Equation of state
    std::string eos = physicsNode["equationOfState"].as<std::string>("idealGas");
    if (eos == "idealGas") {
        params.equationOfState = physics::EquationOfState::IDEAL_GAS;
    } else if (eos == "incompressible") {
        params.equationOfState = physics::EquationOfState::INCOMPRESSIBLE;
    }
    
    return params;
}

std::map<std::string, std::function<Real(const Vector3&)>>
CaseReader::readInitialConditions() {
    std::map<std::string, std::function<Real(const Vector3&)>> conditions;
    
    YAML::Node icNode = config_["initialConditions"];
    if (!icNode) {
        return conditions;
    }
    
    // Velocity
    if (icNode["U"]) {
        Vector3 U0 = parseVector3(icNode["U"]);
        conditions["U"] = [U0](const Vector3&) { return U0.x(); }; // Component-wise
    }
    
    // Pressure
    if (icNode["p"]) {
        Real p0 = icNode["p"].as<Real>();
        conditions["p"] = [p0](const Vector3&) { return p0; };
    }
    
    // Temperature
    if (icNode["T"]) {
        Real T0 = icNode["T"].as<Real>();
        conditions["T"] = [T0](const Vector3&) { return T0; };
    }
    
    // Turbulence
    if (icNode["k"]) {
        Real k0 = icNode["k"].as<Real>();
        conditions["k"] = [k0](const Vector3&) { return k0; };
    }
    
    if (icNode["omega"]) {
        Real omega0 = icNode["omega"].as<Real>();
        conditions["omega"] = [omega0](const Vector3&) { return omega0; };
    }
    
    return conditions;
}

std::map<std::string, std::map<std::string, BoundaryConditionInfo>>
CaseReader::readBoundaryConditions() {
    std::map<std::string, std::map<std::string, BoundaryConditionInfo>> bcMap;
    
    YAML::Node bcNode = config_["boundaryConditions"];
    if (!bcNode) {
        return bcMap;
    }
    
    for (const auto& patch : bcNode) {
        std::string patchName = patch.first.as<std::string>();
        
        for (const auto& field : patch.second) {
            std::string fieldName = field.first.as<std::string>();
            
            if (fieldName == "type") continue; // Skip patch type
            
            BoundaryConditionInfo info;
            
            YAML::Node fieldBC = field.second;
            info.type = fieldBC["type"].as<std::string>();
            
            if (fieldBC["value"]) {
                if (fieldName == "U") {
                    info.vectorValue = parseVector3(fieldBC["value"]);
                } else {
                    info.scalarValue = fieldBC["value"].as<Real>();
                }
            }
            
            if (fieldBC["gradient"]) {
                info.scalarValue = fieldBC["gradient"].as<Real>();
            }
            
            bcMap[patchName][fieldName] = info;
        }
    }
    
    return bcMap;
}

void CaseReader::loadConfiguration() {
    // Look for case configuration file
    std::vector<std::string> configFiles = {
        caseDir_ + "/case.yaml",
        caseDir_ + "/case.yml",
        caseDir_ + "/config.yaml",
        caseDir_ + "/system/controlDict"
    };
    
    bool found = false;
    for (const auto& file : configFiles) {
        if (fs::exists(file)) {
            config_ = YAML::LoadFile(file);
            found = true;
            logger_->info("Loaded configuration from: {}", file);
            break;
        }
    }
    
    if (!found) {
        throw std::runtime_error("No configuration file found in case directory");
    }
}

Vector3 CaseReader::parseVector3(const YAML::Node& node) {
    std::vector<Real> values = node.as<std::vector<Real>>();
    if (values.size() != 3) {
        throw std::runtime_error("Vector must have 3 components");
    }
    return Vector3(values[0], values[1], values[2]);
}
