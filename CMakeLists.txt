cmake_minimum_required(VERSION 3.16)
project(nocta VERSION 0.3.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(NOCTA_BUILD_EXAMPLES "Build examples" ON)
option(NOCTA_ENABLE_SIMD "Enable SIMD optimizations" OFF)

# Compiler flags
if(MSVC)
    add_compile_options(/W3 /D_CRT_SECURE_NO_WARNINGS)
    if(NOCTA_ENABLE_SIMD)
        add_compile_options(/arch:AVX2)
        add_compile_definitions(NOCTA_SIMD_ENABLED)
    endif()
else()
    add_compile_options(-Wall -Wextra)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3)
    else()
        add_compile_options(-g -O0)
    endif()
    if(NOCTA_ENABLE_SIMD)
        add_compile_options(-mavx2 -mfma)
        add_compile_definitions(NOCTA_SIMD_ENABLED)
    endif()
endif()

# Sources - only files we have created
set(NOCTA_SOURCES
    src/core/tensor.c
    src/core/memory.c
    src/core/serialize.c
    src/autograd/node.c
    src/autograd/backward.c
    src/ops/arithmetic.c
    src/ops/matmul.c
    src/ops/activation.c
    src/ops/reduction.c
    src/ops/loss.c
    src/nn/module.c
    src/nn/linear.c
    src/nn/conv.c
    src/nn/batchnorm.c
    src/optim/optimizer.c
    src/optim/sgd.c
    src/optim/adam.c
)

# Static library
add_library(nocta STATIC ${NOCTA_SOURCES})

target_include_directories(nocta
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Math library (not needed on Windows)
if(NOT WIN32)
    target_link_libraries(nocta PUBLIC m)
endif()

# Examples
if(NOCTA_BUILD_EXAMPLES)
    add_executable(example_xor examples/xor.c)
    target_link_libraries(example_xor PRIVATE nocta)
    
    add_executable(example_save_load examples/save_load.c)
    target_link_libraries(example_save_load PRIVATE nocta)
    
    add_executable(example_mnist examples/mnist.c)
    target_link_libraries(example_mnist PRIVATE nocta)
    
    add_executable(example_batchnorm examples/batchnorm.c)
    target_link_libraries(example_batchnorm PRIVATE nocta)
endif()