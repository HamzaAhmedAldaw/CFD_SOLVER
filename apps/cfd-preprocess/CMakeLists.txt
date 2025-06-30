
// === apps/cfd-preprocess/CMakeLists.txt ===
add_executable(cfd-preprocess main.cpp)

target_link_libraries(cfd-preprocess
    PRIVATE
        cfd_core
        yaml-cpp
)

if(CFD_ENABLE_MPI)
    target_link_libraries(cfd-preprocess PRIVATE MPI::MPI_CXX)
endif()

install(TARGETS cfd-preprocess
    RUNTIME DESTINATION bin
)

# Add to CTest
if(CFD_BUILD_TESTS)
    add_test(NAME preprocess_help
        COMMAND cfd-preprocess --help
    )
endif()