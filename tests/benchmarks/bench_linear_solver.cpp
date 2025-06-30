// tests/benchmarks/bench_linear_solver.cpp
#include <benchmark/benchmark.h>
#include "cfd/solvers/LinearSolver.hpp"
#include "cfd/core/Types.hpp"
#include <random>

using namespace cfd;

static void BM_CGSolver(benchmark::State& state) {
    int n = state.range(0);
    
    // Create a symmetric positive definite matrix (Laplacian)
    SparseMatrix A(n, n);
    Vector b(n);
    
    // Fill matrix (1D Laplacian)
    for (int i = 0; i < n; ++i) {
        A.insert(i, i) = 2.0;
        if (i > 0) A.insert(i, i-1) = -1.0;
        if (i < n-1) A.insert(i, i+1) = -1.0;
        b[i] = 1.0;
    }
    A.makeCompressed();
    
    // Create solver
    auto solver = LinearSolver::create("CG");
    solver->setTolerance(1e-6);
    solver->setMaxIterations(1000);
    
    Vector x(n);
    
    for (auto _ : state) {
        x.setZero();
        solver->solve(A, b, x);
        benchmark::DoNotOptimize(x);
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_CGSolver)->Range(64, 4096)->RangeMultiplier(2);

static void BM_BiCGSTABSolver(benchmark::State& state) {
    int n = state.range(0);
    
    // Create a non-symmetric matrix
    SparseMatrix A(n, n);
    Vector b(n);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    // Fill matrix with random entries (ensure diagonal dominance)
    for (int i = 0; i < n; ++i) {
        A.insert(i, i) = 5.0;
        if (i > 0) A.insert(i, i-1) = dis(gen);
        if (i < n-1) A.insert(i, i+1) = dis(gen);
        b[i] = dis(gen);
    }
    A.makeCompressed();
    
    // Create solver
    auto solver = LinearSolver::create("BiCGSTAB");
    solver->setTolerance(1e-6);
    solver->setMaxIterations(1000);
    
    Vector x(n);
    
    for (auto _ : state) {
        x.setZero();
        solver->solve(A, b, x);
        benchmark::DoNotOptimize(x);
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_BiCGSTABSolver)->Range(64, 4096)->RangeMultiplier(2);

BENCHMARK_MAIN();
