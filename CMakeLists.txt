cmake_minimum_required(VERSION 3.16)
project(CFDSolver VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(CFD_ENABLE_MPI "Enable MPI support" ON)
option(CFD_ENABLE_OPENMP "Enable OpenMP support" ON)
option(CFD_USE_DOUBLE_PRECISION "Use double precision" ON)
option(CFD_BUILD_TESTS "Build test suite" ON)
option(CFD_BUILD_EXAMPLES "Build examples" ON)
option(CFD_BUILD_DOCS "Build documentation" OFF)
option(CFD_USE_PETSC "Use PETSc for linear solvers" OFF)
option(CFD_USE_METIS "Use METIS for mesh partitioning" OFF)
option(CFD_USE_CUDA "Enable CUDA support" OFF)
option(CFD_ENABLE_PROFILING "Enable profiling support" OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    
    if(CFD_ENABLE_OPENMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
    
    if(CFD_ENABLE_PROFILING)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    endif()
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    
    if(CFD_ENABLE_OPENMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    endif()
endif()

# Find packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

# Find optional packages
if(CFD_ENABLE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DCFD_ENABLE_MPI)
endif()

if(CFD_USE_PETSC)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    find_package(PETSc REQUIRED)
    add_definitions(-DCFD_USE_PETSC)
endif()

if(CFD_USE_METIS)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    find_package(METIS REQUIRED)
    add_definitions(-DCFD_USE_METIS)
endif()

# Find spdlog
find_package(spdlog REQUIRED)

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)

# Find jsoncpp - try CMake config first (vcpkg), then fall back to pkg-config
find_package(jsoncpp CONFIG QUIET)
if(NOT jsoncpp_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(JSONCPP REQUIRED jsoncpp)
endif()

# Configure precision
if(CFD_USE_DOUBLE_PRECISION)
    add_definitions(-DCFD_DOUBLE_PRECISION)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

# Collect source files
file(GLOB_RECURSE CFD_SOURCES 
    src/*.cpp
)

file(GLOB_RECURSE CFD_HEADERS 
    include/*.hpp
)

# Create library
add_library(cfd_core ${CFD_SOURCES})

# Link libraries
target_link_libraries(cfd_core
    PUBLIC
        Eigen3::Eigen
        Threads::Threads
        spdlog::spdlog
        yaml-cpp
)

# Link jsoncpp properly
if(TARGET JsonCpp::JsonCpp)
    target_link_libraries(cfd_core PUBLIC JsonCpp::JsonCpp)
elseif(JSONCPP_FOUND)
    target_link_libraries(cfd_core PUBLIC ${JSONCPP_LIBRARIES})
    target_include_directories(cfd_core PUBLIC ${JSONCPP_INCLUDE_DIRS})
endif()

# Add MPI if enabled
if(CFD_ENABLE_MPI)
    target_link_libraries(cfd_core PUBLIC MPI::MPI_CXX)
endif()

# Add PETSc if enabled
if(CFD_USE_PETSC)
    target_include_directories(cfd_core PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(cfd_core PUBLIC ${PETSC_LIBRARIES})
endif()

# Add METIS if enabled
if(CFD_USE_METIS)
    target_include_directories(cfd_core PUBLIC ${METIS_INCLUDE_DIRS})
    target_link_libraries(cfd_core PUBLIC ${METIS_LIBRARIES})
endif()

# Build applications
add_subdirectory(apps)

# Build examples
if(CFD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build tests
if(CFD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Build documentation
if(CFD_BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in 
                      ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(doc
            ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM
        )
    endif()
endif()

# Installation
install(TARGETS cfd_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/cfd
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/CFDSolverConfig.cmake.in
    ${CMAKE_BINARY_DIR}/CFDSolverConfig.cmake
    INSTALL_DESTINATION lib/cmake/CFDSolver
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/CFDSolverConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_BINARY_DIR}/CFDSolverConfig.cmake
    ${CMAKE_BINARY_DIR}/CFDSolverConfigVersion.cmake
    DESTINATION lib/cmake/CFDSolver
)

# Summary
message(STATUS "")
message(STATUS "CFD Solver Configuration Summary:")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "  Options:")
message(STATUS "    MPI:            ${CFD_ENABLE_MPI}")
message(STATUS "    OpenMP:         ${CFD_ENABLE_OPENMP}")
message(STATUS "    Double prec.:   ${CFD_USE_DOUBLE_PRECISION}")
message(STATUS "    PETSc:          ${CFD_USE_PETSC}")
message(STATUS "    METIS:          ${CFD_USE_METIS}")
message(STATUS "    CUDA:           ${CFD_USE_CUDA}")
message(STATUS "    Tests:          ${CFD_BUILD_TESTS}")
message(STATUS "    Examples:       ${CFD_BUILD_EXAMPLES}")
message(STATUS "    Documentation:  ${CFD_BUILD_DOCS}")
message(STATUS "")
message(STATUS "  Dependencies:")
message(STATUS "    Eigen3:         ${Eigen3_VERSION}")
message(STATUS "    spdlog:         Found")
message(STATUS "    yaml-cpp:       Found")
if(CFD_ENABLE_MPI)
    message(STATUS "    MPI:            ${MPI_CXX_COMPILER}")
endif()
message(STATUS "")
