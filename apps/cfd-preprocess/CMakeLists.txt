# apps/cfd-preprocess/CMakeLists.txt
add_executable(cfd-preprocess main.cpp)

target_link_libraries(cfd-preprocess
    PRIVATE
        cfd_core
)

install(TARGETS cfd-preprocess
    RUNTIME DESTINATION bin
)
