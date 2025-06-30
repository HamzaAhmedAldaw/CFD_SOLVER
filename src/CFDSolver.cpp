// ===== CORE CFD SOLVER IMPLEMENTATION =====

// === src/CFDSolver.cpp ===
#include "cfd/CFDSolver.hpp"
#include "cfd/io/Logger.hpp"
#include "cfd/parallel/MPI_Wrapper.hpp"
#include "cfd/parallel/Communication.hpp"
#include <filesystem>
#include <chrono>

namespace cfd {

CFDSolver::CFDSolver(const std::string& caseDirectory)
    : caseDir_(caseDirectory), currentTime_(0.0), timeStep_(0) {
    
    // Initialize logger
    logger_ = std::make_shared<io::Logger>("CFDSolver");
    logger_->info("Initializing CFD Solver for case: {}", caseDirectory);
    
    // Check if case directory exists
    if (!std::filesystem::exists(caseDir_)) {
        throw std::runtime_error("Case directory does not exist: " + caseDir_);
    }
}

CFDSolver::~CFDSolver() {
    logger_->info("CFD Solver shutdown");
}

void CFDSolver::initialize() {
    logger_->info("Initializing simulation");
    
    // Read case configuration
    readCase();
    
    // Create fields
    createFields();
    
    // Set initial conditions
    setInitialConditions();
    
    // Set boundary conditions
    setBoundaryConditions();
    
    // Setup numerical schemes and solvers
    setupSolvers();
    
    // Initialize time control
    currentTime_ = settings_.startTime;
    deltaT_ = settings_.deltaT;
    nextWriteTime_ = currentTime_ + settings_.writeInterval;
    
    // Write initial fields
    if (settings_.writeInterval > 0) {
        writeFields();
    }
    
    logger_->info("Initialization complete");
}

void CFDSolver::readCase() {
    logger_->info("Reading case configuration");
    
    caseReader_ = std::make_shared<io::CaseReader>(caseDir_);
    
    // Read mesh
    mesh_ = caseReader_->readMesh();
    logger_->info("Mesh loaded: {} cells, {} faces", 
                  mesh_->numCells(), mesh_->numFaces());
    
    // Read simulation settings
    settings_ = caseReader_->readSimulationSettings();
    
    // Read physics parameters
    auto physicsParams = caseReader_->readPhysicsParameters();
    
    // Create physics models
    nsEquations_ = std::make_shared<physics::NavierStokes>(physicsParams);
    
    if (settings_.turbulence) {
        turbulence_ = physics::TurbulenceModel::create(
            settings_.turbulenceModel, mesh_, physicsParams);
    }
    
    thermo_ = std::make_shared<physics::Thermodynamics>(
        physicsParams.equationOfState, physicsParams);
}

void CFDSolver::createFields() {
    logger_->info("Creating solution fields");
    
    // Primary variables
    U_ = std::make_shared<VectorField>(mesh_, "U");
    p_ = std::make_shared<ScalarField>(mesh_, "p");
    
    // Density
    rho_ = std::make_shared<ScalarField>(mesh_, "rho");
    
    // Temperature (for compressible or heat transfer)
    if (settings_.solverType == SolverType::COMPRESSIBLE ||
        settings_.solverType == SolverType::LOW_MACH) {
        T_ = std::make_shared<ScalarField>(mesh_, "T");
    }
    
    // Material properties
    mu_ = std::make_shared<ScalarField>(mesh_, "mu");
    
    // Turbulence fields
    if (settings_.turbulence) {
        k_ = std::make_shared<ScalarField>(mesh_, "k");
        
        if (settings_.turbulenceModel == TurbulenceModelType::K_OMEGA ||
            settings_.turbulenceModel == TurbulenceModelType::K_OMEGA_SST) {
            omega_ = std::make_shared<ScalarField>(mesh_, "omega");
        } else if (settings_.turbulenceModel == TurbulenceModelType::K_EPSILON) {
            // epsilon_ would be created here
        }
        
        nut_ = std::make_shared<ScalarField>(mesh_, "nut");
        alphaEff_ = std::make_shared<ScalarField>(mesh_, "alphaEff");
    }
}

void CFDSolver::setInitialConditions() {
    logger_->info("Setting initial conditions");
    
    auto initialConditions = caseReader_->readInitialConditions();
    
    // Velocity
    if (initialConditions.count("U")) {
        U_->initialize(initialConditions["U"]);
    }
    
    // Pressure
    if (initialConditions.count("p")) {
        p_->initialize(initialConditions["p"]);
    }
    
    // Temperature
    if (T_ && initialConditions.count("T")) {
        T_->initialize(initialConditions["T"]);
    }
    
    // Density (compute from equation of state if needed)
    if (settings_.solverType == SolverType::INCOMPRESSIBLE) {
        *rho_ = settings_.referencePressure / (287.0 * settings_.referenceTemperature);
    } else {
        // Compute from EOS
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            (*rho_)[i] = thermo_->density((*p_)[i], (*T_)[i]);
        }
    }
    
    // Turbulence
    if (settings_.turbulence) {
        if (initialConditions.count("k")) {
            k_->initialize(initialConditions["k"]);
        }
        if (omega_ && initialConditions.count("omega")) {
            omega_->initialize(initialConditions["omega"]);
        }
    }
    
    // Update material properties
    updateProperties();
}

void CFDSolver::setBoundaryConditions() {
    logger_->info("Setting boundary conditions");
    
    auto boundaryConditions = caseReader_->readBoundaryConditions();
    
    for (const auto& [patchName, patchBCs] : boundaryConditions) {
        logger_->debug("Setting BCs for patch: {}", patchName);
        
        // Velocity BCs
        if (patchBCs.count("U")) {
            U_->addBoundaryCondition(patchName, patchBCs.at("U"));
        }
        
        // Pressure BCs
        if (patchBCs.count("p")) {
            p_->addBoundaryCondition(patchName, patchBCs.at("p"));
        }
        
        // Temperature BCs
        if (T_ && patchBCs.count("T")) {
            T_->addBoundaryCondition(patchName, patchBCs.at("T"));
        }
        
        // Turbulence BCs
        if (settings_.turbulence) {
            if (patchBCs.count("k")) {
                k_->addBoundaryCondition(patchName, patchBCs.at("k"));
            }
            if (omega_ && patchBCs.count("omega")) {
                omega_->addBoundaryCondition(patchName, patchBCs.at("omega"));
            }
        }
    }
    
    // Apply boundary conditions
    U_->updateBoundaryConditions();
    p_->updateBoundaryConditions();
    if (T_) T_->updateBoundaryConditions();
    if (k_) k_->updateBoundaryConditions();
    if (omega_) omega_->updateBoundaryConditions();
}

void CFDSolver::setupSolvers() {
    logger_->info("Setting up numerical schemes and solvers");
    
    // Gradient scheme
    auto gradSchemeType = caseReader_->readNumericsSettings().gradientScheme;
    gradScheme_ = numerics::GradientScheme::create(gradSchemeType, mesh_);
    
    // Time integration
    timeIntegration_ = numerics::TimeIntegration::create(
        settings_.timeScheme, mesh_);
    
    // Flux scheme (for compressible)
    if (settings_.solverType != SolverType::INCOMPRESSIBLE) {
        auto fluxSchemeType = caseReader_->readNumericsSettings().fluxScheme;
        fluxScheme_ = numerics::FluxScheme<CompressibleState>::create(
            fluxSchemeType, mesh_);
    }
    
    // Pressure-velocity coupling
    auto pvSettings = caseReader_->readNumericsSettings().pressureVelocityCoupling;
    pvCoupling_ = std::make_shared<solvers::PressureVelocityCoupling>(
        mesh_, pvSettings);
    
    // Linear solver
    auto linearSolverSettings = caseReader_->readSolverSettings();
    linearSolver_ = solvers::LinearSolver::create(
        linearSolverSettings.type, linearSolverSettings);
    
    // Nonlinear solver (for implicit schemes)
    if (settings_.timeScheme != TimeScheme::EULER_EXPLICIT &&
        settings_.timeScheme != TimeScheme::RK4) {
        nonlinearSolver_ = std::make_shared<solvers::NonlinearSolver>(
            mesh_, linearSolver_);
    }
    
    // Setup communication pattern for parallel runs
    if (parallel::MPIWrapper::isParallel()) {
        auto commPattern = std::make_shared<parallel::CommunicationPattern>(*mesh_);
        commPattern->build();
        
        // Create field communicators
        // fieldComm_ = std::make_shared<parallel::FieldCommunicator<Real>>(*commPattern);
    }
}

void CFDSolver::runSteady() {
    logger_->info("Running steady-state simulation");
    
    int iteration = 0;
    bool converged = false;
    
    while (!converged && iteration < settings_.maxIterations) {
        iteration++;
        
        logger_->info("Iteration {}", iteration);
        
        // Solve one pseudo-time step
        solveTimeStep();
        
        // Check convergence
        converged = residuals_.checkConvergence(settings_.convergenceTolerance);
        
        // Print residuals
        if (iteration % 10 == 0 || converged) {
            residuals_.print();
        }
        
        // Write fields if needed
        if (settings_.writeFrequency > 0 && iteration % settings_.writeFrequency == 0) {
            writeFields();
        }
    }
    
    if (converged) {
        logger_->info("Solution converged in {} iterations", iteration);
    } else {
        logger_->warning("Maximum iterations reached without convergence");
    }
    
    // Write final fields
    writeFields();
}

void CFDSolver::solveTimeStep() {
    // Update material properties
    updateProperties();
    
    // Store old residuals for monitoring
    auto oldResiduals = residuals_.current;
    
    // Solve based on solver type
    switch (settings_.solverType) {
        case SolverType::INCOMPRESSIBLE:
            solveIncompressible();
            break;
        case SolverType::COMPRESSIBLE:
            solveCompressible();
            break;
        case SolverType::LOW_MACH:
            solveLowMach();
            break;
    }
    
    // Solve turbulence if enabled
    if (settings_.turbulence) {
        solveTurbulence();
        updateTurbulentViscosity();
    }
    
    // Update residuals
    for (const auto& [field, residual] : residuals_.current) {
        residuals_.update(field, residual);
    }
}

void CFDSolver::solveIncompressible() {
    // SIMPLE/PISO/PIMPLE algorithm
    
    // Under-relaxation factors
    const Real alphaU = 0.7;
    const Real alphaP = 0.3;
    
    // Store old values for under-relaxation
    VectorField Uold = *U_;
    ScalarField pold = *p_;
    
    // Momentum predictor
    {
        // Assemble momentum equation
        SparseMatrix A;
        VectorX b, x;
        
        // Build matrix using automatic differentiation
        assembleVectorMatrix(*U_, A, b, [this](const VectorField& U) {
            return computeMomentumResidual(U);
        });
        
        // Solve
        linearSolver_->solve(A, b, x);
        
        // Update velocity with under-relaxation
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            (*U_)[i] = alphaU * vectorFromSolution(x, i) + (1 - alphaU) * Uold[i];
        }
    }
    
    // Pressure correction (PISO loop)
    for (int corr = 0; corr < pvCoupling_->numCorrectors(); ++corr) {
        // Solve pressure equation
        ScalarField pCorr(*mesh_, "pCorr");
        pvCoupling_->solvePressureCorrection(*U_, *p_, pCorr);
        
        // Correct pressure
        *p_ += alphaP * pCorr;
        
        // Correct velocity
        pvCoupling_->correctVelocity(*U_, pCorr);
        
        // Update boundary conditions
        U_->updateBoundaryConditions();
        p_->updateBoundaryConditions();
    }
    
    // Compute residuals
    residuals_.current["U"] = computeResidualNorm(*U_, Uold, deltaT_);
    residuals_.current["p"] = computeResidualNorm(*p_, pold, deltaT_);
}

void CFDSolver::solveCompressible() {
    // Density-based coupled solver
    
    // Conservative variables
    std::vector<ScalarField> Q(5);
    Q[0] = *rho_;                              // Density
    Q[1] = ScalarField(*mesh_, "rhoU");        // Momentum x
    Q[2] = ScalarField(*mesh_, "rhoV");        // Momentum y  
    Q[3] = ScalarField(*mesh_, "rhoW");        // Momentum z
    Q[4] = ScalarField(*mesh_, "rhoE");        // Total energy
    
    // Convert primitive to conservative
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        Q[1][i] = (*rho_)[i] * (*U_)[i].x();
        Q[2][i] = (*rho_)[i] * (*U_)[i].y();
        Q[3][i] = (*rho_)[i] * (*U_)[i].z();
        
        Real kinetic = 0.5 * (*U_)[i].squaredNorm();
        Real internal = nsEquations_->internalEnergy((*p_)[i], (*rho_)[i]);
        Q[4][i] = (*rho_)[i] * (kinetic + internal);
    }
    
    // Time integration
    if (settings_.timeScheme == TimeScheme::EULER_EXPLICIT) {
        // Explicit time stepping
        std::vector<ScalarField> R(5);
        computeCompressibleResidual(Q, R);
        
        // Update
        Real dt = deltaT_;
        for (int eq = 0; eq < 5; ++eq) {
            for (Index i = 0; i < mesh_->numCells(); ++i) {
                Q[eq][i] -= dt / mesh_->cell(i).volume() * R[eq][i];
            }
        }
    } else {
        // Implicit time stepping with Newton-Krylov
        nonlinearSolver_->solve(Q, [this](const auto& Q) {
            std::vector<ScalarField> R(5);
            computeCompressibleResidual(Q, R);
            return R;
        });
    }
    
    // Convert conservative to primitive
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        (*rho_)[i] = Q[0][i];
        (*U_)[i].x() = Q[1][i] / (*rho_)[i];
        (*U_)[i].y() = Q[2][i] / (*rho_)[i];
        (*U_)[i].z() = Q[3][i] / (*rho_)[i];
        
        Real kinetic = 0.5 * (*U_)[i].squaredNorm();
        Real internal = Q[4][i] / (*rho_)[i] - kinetic;
        (*p_)[i] = nsEquations_->pressure((*rho_)[i], internal);
        
        if (T_) {
            (*T_)[i] = thermo_->temperature((*p_)[i], (*rho_)[i]);
        }
    }
    
    // Update residuals
    // (Implementation details omitted for brevity)
}

void CFDSolver::solveTurbulence() {
    if (!turbulence_) return;
    
    // Solve turbulence transport equations
    turbulence_->solve(*U_, *nut_, deltaT_);
    
    // Update turbulent viscosity
    updateTurbulentViscosity();
    
    // Update effective diffusivity
    if (alphaEff_) {
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            // alphaEff = alpha + alphat
            Real alpha = (*mu_)[i] / ((*rho_)[i] * nsEquations_->Pr());
            Real alphat = (*nut_)[i] / nsEquations_->Prt();
            (*alphaEff_)[i] = alpha + alphat;
        }
    }
}

void CFDSolver::updateProperties() {
    // Update density (for compressible)
    if (settings_.solverType != SolverType::INCOMPRESSIBLE) {
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            (*rho_)[i] = thermo_->density((*p_)[i], T_ ? (*T_)[i] : settings_.referenceTemperature);
        }
    }
    
    // Update viscosity
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        Real T = T_ ? (*T_)[i] : settings_.referenceTemperature;
        (*mu_)[i] = nsEquations_->viscosity(T);
    }
}

void CFDSolver::updateTurbulentViscosity() {
    if (!turbulence_) return;
    
    // Compute turbulent viscosity from turbulence variables
    turbulence_->updateNut(*k_, omega_ ? *omega_ : nullptr, *nut_);
    
    // Limit turbulent viscosity
    const Real nutMax = 1e5 * (*mu_)[0]; // Large but finite limit
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        (*nut_)[i] = std::min((*nut_)[i], nutMax);
        (*nut_)[i] = std::max((*nut_)[i], Real(0));
    }
}

void CFDSolver::updateTimeStep() {
    if (!settings_.adjustTimeStep) return;
    
    // Compute maximum Courant number
    Real maxCo = 0.0;
    
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        Real localCo = 0.0;
        
        for (const auto& face : mesh_->cell(i).faces()) {
            Real phi = (*U_)[i].dot(face.normal()) * face.area();
            localCo += std::abs(phi) * deltaT_ / mesh_->cell(i).volume();
        }
        
        maxCo = std::max(maxCo, localCo);
    }
    
    // Global maximum for parallel runs
    if (parallel::MPIWrapper::isParallel()) {
        maxCo = parallel::globalMax(maxCo);
    }
    
    // Adjust time step
    if (maxCo > SMALL) {
        Real factor = settings_.maxCo / maxCo;
        
        // Limit rate of change
        factor = std::min(factor, Real(1.2));
        factor = std::max(factor, Real(0.5));
        
        deltaT_ *= factor;
        deltaT_ = std::min(deltaT_, settings_.maxDeltaT);
    }
    
    logger_->debug("Max Courant number: {:.3f}, deltaT: {:.3e}", maxCo, deltaT_);
}

bool CFDSolver::writeNow() const {
    if (settings_.writeFrequency > 0) {
        return timeStep_ % settings_.writeFrequency == 0;
    } else {
        return currentTime_ >= nextWriteTime_;
    }
}

void CFDSolver::writeFields() {
    logger_->info("Writing fields at time = {}", currentTime_);
    
    if (!vtkWriter_) {
        vtkWriter_ = std::make_shared<io::VTKWriter>(mesh_, caseDir_ + "/VTK");
    }
    
    // Collect fields to write
    std::vector<std::pair<std::string, SharedPtr<FieldBase>>> fields;
    
    fields.emplace_back("U", U_);
    fields.emplace_back("p", p_);
    fields.emplace_back("rho", rho_);
    
    if (T_) fields.emplace_back("T", T_);
    if (k_) fields.emplace_back("k", k_);
    if (omega_) fields.emplace_back("omega", omega_);
    if (nut_) fields.emplace_back("nut", nut_);
    
    // Write VTK file
    vtkWriter_->writeTimeStep(currentTime_, fields);
    
    // Write restart data
    // (Implementation details omitted)
}

void CFDSolver::writeResiduals() {
    std::string filename = caseDir_ + "/residuals.dat";
    std::ofstream file(filename, std::ios::app);
    
    if (timeStep_ == 1) {
        // Write header
        file << "# Time";
        for (const auto& [field, _] : residuals_.current) {
            file << " " << field;
        }
        file << "\n";
    }
    
    // Write data
    file << currentTime_;
    for (const auto& [field, residual] : residuals_.current) {
        file << " " << std::scientific << residual;
    }
    file << "\n";
}

// Helper functions
VectorField CFDSolver::computeMomentumResidual(const VectorField& U) {
    VectorField residual(*mesh_, "residual");
    
    // Convection term
    auto convection = nsEquations_->convection(U, U);
    
    // Diffusion term
    auto diffusion = nsEquations_->diffusion(U, *mu_, nut_.get());
    
    // Pressure gradient
    auto pressureGrad = gradScheme_->gradient(*p_);
    
    // Source terms (gravity, etc.)
    VectorField source(*mesh_, "source");
    if (nsEquations_->hasGravity()) {
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            source[i] = (*rho_)[i] * nsEquations_->gravity();
        }
    }
    
    // Assemble residual
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        residual[i] = convection[i] - diffusion[i] + pressureGrad[i] - source[i];
    }
    
    return residual;
}

void CFDSolver::computeCompressibleResidual(
    const std::vector<ScalarField>& Q,
    std::vector<ScalarField>& R) {
    
    // Zero residuals
    for (auto& r : R) {
        r = 0.0;
    }
    
    // Loop over faces
    for (Index faceId = 0; faceId < mesh_->numFaces(); ++faceId) {
        const Face& face = mesh_->face(faceId);
        
        // Get left and right states
        physics::CompressibleState stateL, stateR;
        
        // Reconstruct states at face
        // (Implementation of high-order reconstruction omitted)
        
        // Compute flux
        auto flux = fluxScheme_->compute(stateL, stateR, face.normal());
        
        // Add to residuals
        Index owner = face.owner();
        for (int eq = 0; eq < 5; ++eq) {
            R[eq][owner] += flux[eq] * face.area();
        }
        
        if (!face.isBoundary()) {
            Index neighbor = face.neighbor();
            for (int eq = 0; eq < 5; ++eq) {
                R[eq][neighbor] -= flux[eq] * face.area();
            }
        }
    }
    
    // Add viscous fluxes
    // (Implementation omitted)
}

void CFDSolver::assembleVectorMatrix(
    const VectorField& field,
    SparseMatrix& A,
    VectorX& b,
    std::function<VectorField(const VectorField&)> residualFunc) {
    
    const Index n = mesh_->numCells();
    const Index dof = 3 * n; // 3 components per cell
    
    A.resize(dof, dof);
    b.resize(dof);
    
    // Use automatic differentiation to compute Jacobian
    ad::FieldAD<Vector3> fieldAD;
    
    std::vector<Triplet> triplets;
    triplets.reserve(dof * 7); // Estimate
    
    // For each cell and component
    for (Index cellId = 0; cellId < n; ++cellId) {
        for (int comp = 0; comp < 3; ++comp) {
            Index rowIdx = 3 * cellId + comp;
            
            // Create dual field with seed at this DOF
            auto dualField = fieldAD.toDualField(field, rowIdx);
            
            // Compute residual with AD
            auto residual = residualFunc(*dualField);
            
            // Extract Jacobian entries
            for (Index j = 0; j < n; ++j) {
                for (int c = 0; c < 3; ++c) {
                    Index colIdx = 3 * j + c;
                    Real deriv = residual[j][c].derivative();
                    
                    if (std::abs(deriv) > EPSILON) {
                        triplets.emplace_back(rowIdx, colIdx, deriv);
                    }
                }
            }
            
            // RHS is negative residual at current state
            b[rowIdx] = -residualFunc(field)[cellId][comp];
        }
    }
    
    A.setFromTriplets(triplets.begin(), triplets.end());
}

Real CFDSolver::computeResidualNorm(const FieldBase& newField,
                                   const FieldBase& oldField,
                                   Real dt) {
    Real norm = 0.0;
    Real normFactor = 0.0;
    
    if (auto scalarNew = dynamic_cast<const ScalarField*>(&newField)) {
        auto scalarOld = dynamic_cast<const ScalarField*>(&oldField);
        
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            Real res = ((*scalarNew)[i] - (*scalarOld)[i]) / dt;
            norm += res * res * mesh_->cell(i).volume();
            normFactor += (*scalarNew)[i] * (*scalarNew)[i] * mesh_->cell(i).volume();
        }
    } else if (auto vectorNew = dynamic_cast<const VectorField*>(&newField)) {
        auto vectorOld = dynamic_cast<const VectorField*>(&oldField);
        
        for (Index i = 0; i < mesh_->numCells(); ++i) {
            Vector3 res = ((*vectorNew)[i] - (*vectorOld)[i]) / dt;
            norm += res.squaredNorm() * mesh_->cell(i).volume();
            normFactor += (*vectorNew)[i].squaredNorm() * mesh_->cell(i).volume();
        }
    }
    
    // Global sum for parallel
    if (parallel::MPIWrapper::isParallel()) {
        norm = parallel::globalSum(norm);
        normFactor = parallel::globalSum(normFactor);
    }
    
    return std::sqrt(norm / (normFactor + SMALL));
}

Vector3 CFDSolver::vectorFromSolution(const VectorX& x, Index cellId) {
    return Vector3(x[3*cellId], x[3*cellId+1], x[3*cellId+2]);
}

// ResidualMonitor implementation
void CFDSolver::ResidualMonitor::update(const std::string& field, Real residual) {
    if (initial.find(field) == initial.end()) {
        initial[field] = residual;
    }
    
    current[field] = residual;
    history[field].push_back(residual);
    
    // Limit history size
    if (history[field].size() > 1000) {
        history[field].erase(history[field].begin());
    }
}

bool CFDSolver::ResidualMonitor::checkConvergence(Real tolerance) const {
    for (const auto& [field, residual] : current) {
        if (initial.find(field) == initial.end()) continue;
        
        Real relativeResidual = residual / (initial.at(field) + SMALL);
        if (relativeResidual > tolerance) {
            return false;
        }
    }
    return true;
}

void CFDSolver::ResidualMonitor::print() const {
    std::stringstream ss;
    ss << "Residuals:";
    
    for (const auto& [field, residual] : current) {
        Real relative = 1.0;
        if (initial.find(field) != initial.end()) {
            relative = residual / (initial.at(field) + SMALL);
        }
        ss << " " << field << "=" << std::scientific << std::setprecision(3) 
           << residual << " (" << relative << ")";
    }
    
    io::Logger::instance().info(ss.str());
}

} // namespace cfd
