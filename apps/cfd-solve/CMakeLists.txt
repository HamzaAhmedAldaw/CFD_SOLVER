// === apps/cfd-solve/CMakeLists.txt ===
add_executable(cfd-solve main.cpp)

target_link_libraries(cfd-solve
    PRIVATE
        cfd_core
)

if(CFD_ENABLE_MPI)
    target_link_libraries(cfd-solve PRIVATE MPI::MPI_CXX)
endif()

install(TARGETS cfd-solve
    RUNTIME DESTINATION bin
)

# Add to CTest
if(CFD_BUILD_TESTS)
    add_test(NAME solve_help
        COMMAND cfd-solve --help
    )
endif()

# Create wrapper script for MPI execution
if(CFD_ENABLE_MPI)
    file(WRITE ${CMAKE_BINARY_DIR}/bin/cfd-solve-mpi
"#!/bin/bash
# MPI wrapper for cfd-solve
# Usage: cfd-solve-mpi <num_procs> <case_directory>

if [ \$# -lt 2 ]; then
    echo \"Usage: \$0 <num_procs> <case_directory>\"
    exit 1
fi

NPROCS=\$1
shift

mpirun -np \$NPROCS ${CMAKE_INSTALL_PREFIX}/bin/cfd-solve -p \"\$@\"
"
    )
    
    file(CHMOD ${CMAKE_BINARY_DIR}/bin/cfd-solve-mpi
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
    )
    
    install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/cfd-solve-mpi
        DESTINATION bin
    )
endif()