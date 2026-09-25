cmake_minimum_required(VERSION 3.16)
project(hpc_vector_norm C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HPC_NATIVE_ARCH "Tune for the build host (-march=native, enables AVX-512 when present)" ON)

find_package(OpenMP REQUIRED)

# Optimization flags
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pedantic")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(HPC_NATIVE_ARCH AND COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    elseif(COMPILER_SUPPORTS_AVX2)
        # Portable x86 baseline: AVX2 + FMA + F16C (Haswell and newer)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c")
    endif()
endif()

add_library(hpc_vector_norm STATIC
    src/hpc_vector_norm.cpp
)
target_include_directories(hpc_vector_norm
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(hpc_vector_norm PUBLIC OpenMP::OpenMP_CXX)

add_executable(hpc_vector_norm_bench
    src/benchmark.cpp
)
target_include_directories(hpc_vector_norm_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hpc_vector_norm_bench PRIVATE hpc_vector_norm)

add_executable(hpc_vector_norm_test
    tests/test_correctness.cpp
)
target_include_directories(hpc_vector_norm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hpc_vector_norm_test PRIVATE hpc_vector_norm)

add_executable(test_dsa_runtime
    tests/test_dsa_runtime.cpp
)
target_include_directories(test_dsa_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_dsa_runtime PRIVATE OpenMP::OpenMP_CXX)

enable_testing()
add_test(NAME correctness COMMAND hpc_vector_norm_test)
add_test(NAME dsa_runtime_sanitizer COMMAND test_dsa_runtime)

