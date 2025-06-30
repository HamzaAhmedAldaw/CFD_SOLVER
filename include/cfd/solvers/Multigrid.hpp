#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Field.hpp"
#include "cfd/core/Mesh.hpp"
#include "cfd/solvers/LinearSolver.hpp"
#include <vector>
#include <memory>

namespace cfd::solvers {

// Multigrid cycle types
enum class MGCycleType {
    V_CYCLE,
    W_CYCLE,
    F_CYCLE,
    FMG_CYCLE  // Full Multigrid
};

// Base multigrid solver class
class MultigridSolver : public LinearSolver {
public:
    struct Settings : LinearSolver::Settings {
        // Multigrid specific settings
        MGCycleType cycleType = MGCycleType::V_CYCLE;
        int maxLevels = 5;
        int preSmoothingSteps = 2;
        int postSmoothingSteps = 2;
        int coarsestGridSweeps = 20;
        Real coarseningRatio = 0.5;
        
        // Smoother settings
        LinearSolverType smootherType = LinearSolverType::GMRES;
        Real smootherRelaxation = 0.7;
        
        // Convergence
        Real mgTolerance = 1e-6;
        int maxCycles = 20;
    };
    
    MultigridSolver(const Mesh& fineMesh, const Settings& settings = Settings());
    
    LinearSolverResult solve(const SparseMatrix& A,
                           const VectorX& b,
                           VectorX& x) override;
    
protected:
    const Mesh& fineMesh_;
    Settings mgSettings_;
    
    // Grid hierarchy
    struct GridLevel {
        SharedPtr<Mesh> mesh;
        SparseMatrix A;
        SparseMatrix R;  // Restriction operator
        SparseMatrix P;  // Prolongation operator
        SharedPtr<LinearSolver> smoother;
        
        // Working vectors
        VectorX x, b, r;
    };
    
    std::vector<GridLevel> levels_;
    
    // Setup functions
    virtual void setupHierarchy(const SparseMatrix& A);
    virtual void createCoarseMesh(const Mesh& fineMesh, Mesh& coarseMesh);
    virtual void setupTransferOperators(int level);
    virtual void setupSmoother(int level);
    
    // Multigrid operations
    void cycle(int level);
    void vCycle(int level);
    void wCycle(int level);
    void fCycle(int level);
    void fullMultigrid();
    
    // Core operations
    void smooth(int level, int iterations);
    void restrict(int fineLevel, const VectorX& fine, VectorX& coarse);
    void prolongate(int fineLevel, const VectorX& coarse, VectorX& fine);
    void computeResidual(int level);
    
    // Coarse grid solver
    void solveCoarseGrid(int level);
};

// Geometric multigrid for structured grids
class GeometricMultigrid : public MultigridSolver {
public:
    GeometricMultigrid(const Mesh& fineMesh, const Settings& settings = Settings())
        : MultigridSolver(fineMesh, settings) {}
    
protected:
    void createCoarseMesh(const Mesh& fineMesh, Mesh& coarseMesh) override;
    void setupTransferOperators(int level) override;
    
private:
    // Structured coarsening
    void coarsenStructured(const Mesh& fineMesh, Mesh& coarseMesh);
    
    // Standard restriction/prolongation for structured grids
    void setupStructuredTransfer(int level);
};

// Algebraic multigrid (AMG)
class AlgebraicMultigrid : public MultigridSolver {
public:
    struct AMGSettings : Settings {
        // Coarsening parameters
        Real strongThreshold = 0.25;
        Real maxRowSum = 0.9;
        int aggressiveCoarseningLevels = 0;
        
        // Interpolation
        enum InterpolationType {
            CLASSICAL,
            DIRECT,
            STANDARD,
            EXTENDED
        } interpolationType = STANDARD;
        
        // Smoothing
        bool useGaussSeidel = true;
        bool symmetricSmoothing = false;
    };
    
    AlgebraicMultigrid(const SparseMatrix& A, 
                       const AMGSettings& settings = AMGSettings());
    
protected:
    AMGSettings amgSettings_;
    
    void setupHierarchy(const SparseMatrix& A) override;
    
private:
    // AMG components
    void performCoarsening(int level);
    void computeStrongConnections(int level);
    void selectCoarseNodes(int level);
    void setupInterpolation(int level);
    void computeGalerkinProduct(int level);
    
    // Strength of connection
    struct StrengthMatrix {
        std::vector<std::set<int>> strong;
        std::vector<std::set<int>> strongTranspose;
    };
    
    std::vector<StrengthMatrix> strength_;
    std::vector<std::vector<int>> coarseMapping_;
    
    // Coarsening algorithms
    void classicalCoarsening(int level);
    void aggressiveCoarsening(int level);
    void PMIS(int level);  // Parallel Modified Independent Set
    
    // Interpolation construction
    void classicalInterpolation(int level);
    void directInterpolation(int level);
    void standardInterpolation(int level);
    void extendedInterpolation(int level);
};

// Multigrid preconditioner wrapper
class MultigridPreconditioner : public Preconditioner {
public:
    MultigridPreconditioner(SharedPtr<MultigridSolver> mgSolver)
        : mgSolver_(mgSolver) {}
    
    VectorX apply(const VectorX& r) const override;
    void setup(const SparseMatrix& A) override;
    
private:
    SharedPtr<MultigridSolver> mgSolver_;
};

// Field-based multigrid solver
template<typename T>
class FieldMultigrid {
public:
    FieldMultigrid(const std::vector<SharedPtr<Mesh>>& meshHierarchy)
        : meshHierarchy_(meshHierarchy) {
        setupFieldHierarchy();
    }
    
    // Solve field equation
    void solve(Field<T>& phi, const Field<T>& source,
              std::function<void(const Field<T>&, Field<T>&)> operatorFunc);
    
private:
    std::vector<SharedPtr<Mesh>> meshHierarchy_;
    std::vector<SharedPtr<Field<T>>> fieldHierarchy_;
    std::vector<SharedPtr<Field<T>>> sourceHierarchy_;
    std::vector<SharedPtr<Field<T>>> residualHierarchy_;
    
    void setupFieldHierarchy();
    void restrictField(const Field<T>& fine, Field<T>& coarse);
    void prolongateField(const Field<T>& coarse, Field<T>& fine);
    void smoothField(Field<T>& phi, const Field<T>& source,
                    std::function<void(const Field<T>&, Field<T>&)> operatorFunc,
                    int iterations);
};

// Multigrid for systems (block matrices)
template<int BlockSize>
class BlockMultigrid : public MultigridSolver {
public:
    BlockMultigrid(const Mesh& fineMesh, const Settings& settings = Settings())
        : MultigridSolver(fineMesh, settings) {}
    
protected:
    // Block-aware operations
    void setupBlockTransferOperators(int level);
    void blockSmooth(int level, int iterations);
    
private:
    using BlockMatrix = Eigen::Matrix<Real, BlockSize, BlockSize>;
    using BlockVector = Eigen::Matrix<Real, BlockSize, 1>;
    
    // Convert between block and scalar representations
    void scalarToBlock(const VectorX& scalar, 
                      std::vector<BlockVector>& blocks) const;
    void blockToScalar(const std::vector<BlockVector>& blocks,
                      VectorX& scalar) const;
};

// Additive multigrid (parallel smoother)
class AdditiveMultigrid : public MultigridSolver {
public:
    using MultigridSolver::MultigridSolver;
    
protected:
    void smooth(int level, int iterations) override;
    
private:
    // Parallel smoothing on all levels simultaneously
    void additiveSmooth(VectorX& x, const VectorX& b);
};

// Adaptive multigrid
class AdaptiveMultigrid : public MultigridSolver {
public:
    AdaptiveMultigrid(const Mesh& fineMesh, const Settings& settings = Settings())
        : MultigridSolver(fineMesh, settings) {}
    
    // Adapt hierarchy based on solution
    void adaptHierarchy(const VectorX& solution, const VectorX& residual);
    
protected:
    // Adaptive coarsening based on error indicators
    void adaptiveCoarsening(const VectorX& errorIndicator);
    
private:
    std::vector<Real> errorIndicators_;
    std::vector<bool> refinementFlags_;
};

// Factory functions
inline SharedPtr<MultigridSolver> createMultigridSolver(
    const std::string& type,
    const Mesh& mesh,
    const MultigridSolver::Settings& settings = MultigridSolver::Settings()) {
    
    if (type == "geometric") {
        return std::make_shared<GeometricMultigrid>(mesh, settings);
    } else if (type == "algebraic") {
        return std::make_shared<AlgebraicMultigrid>(
            SparseMatrix(), AlgebraicMultigrid::AMGSettings());
    } else if (type == "additive") {
        return std::make_shared<AdditiveMultigrid>(mesh, settings);
    } else if (type == "adaptive") {
        return std::make_shared<AdaptiveMultigrid>(mesh, settings);
    } else {
        throw std::runtime_error("Unknown multigrid type: " + type);
    }
}

} // namespace cfd::solvers