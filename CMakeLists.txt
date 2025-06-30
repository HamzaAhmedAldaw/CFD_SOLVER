cmake_minimum_required(VERSION 3.20)
project(CFDSolver VERSION 1.0.0 LANGUAGES CXX)

# Set C++17 as required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find packages
find_package(yaml-cpp REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(OpenMP)
find_package(MPI)

# Core library sources
set(CORE_SOURCES
    src/CFDSolver.cpp
    src/ad/AutoDiff.cpp
    src/ad/DualNumber.cpp
    src/ad/TapeRecorder.cpp
    src/core/BoundaryCondition.cpp
    src/core/Cell.cpp
    src/core/Face.cpp
    src/core/Field.cpp
    src/core/Mesh.cpp
    src/io/CaseReader.cpp
    src/io/VTKWriter.cpp
    src/memory/MemoryPool.cpp
    src/numerics/FluxScheme.cpp
    src/numerics/GradientScheme.cpp
    src/numerics/Interpolation.cpp
    src/numerics/Limiter.cpp
    src/numerics/TimeIntegration.cpp
    src/parallel/Communication.cpp
    src/parallel/Domain.cpp
    src/physics/NavierStokes.cpp
    src/physics/Thermodynamics.cpp
    src/physics/TurbulenceModel.cpp
    src/solvers/LinearSolver.cpp
    src/solvers/Multigrid.cpp
    src/solvers/NonlinearSolver.cpp
    src/solvers/PressureVelocityCoupling.cpp
)

# Create core library
add_library(cfd_core ${CORE_SOURCES})

target_include_directories(cfd_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cfd_core PUBLIC
    yaml-cpp::yaml-cpp
    JsonCpp::JsonCpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cfd_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MPI_FOUND)
    target_link_libraries(cfd_core PUBLIC MPI::MPI_CXX)
endif()

# Applications
add_subdirectory(apps)

# Installation
install(TARGETS cfd_core
    EXPORT CFDSolverTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/cfd DESTINATION include)
