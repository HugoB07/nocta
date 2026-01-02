cmake_minimum_required(VERSION 3.18)
project(nocta VERSION 0.5.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(NOCTA_BUILD_EXAMPLES "Build examples" ON)
option(NOCTA_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(NOCTA_ENABLE_OPENMP "Enable OpenMP multi-threading" ON)
option(NOCTA_ENABLE_CUDA "Enable CUDA GPU backend" OFF)

# ============================================
# CUDA Support
# ============================================
if(NOCTA_ENABLE_CUDA)
    # Set CUDA architectures BEFORE enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    endif()
    
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    add_compile_definitions(NOCTA_CUDA_ENABLED)
    message(STATUS "CUDA enabled. Found: ${CUDAToolkit_VERSION}")
endif()

# Find OpenMP
if(NOCTA_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        add_compile_definitions(NOCTA_OPENMP_ENABLED)
    else()
        message(WARNING "OpenMP not found, disabling multi-threading")
        set(NOCTA_ENABLE_OPENMP OFF)
    endif()
endif()

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
        add_compile_options(-O3 -march=native -ffast-math -funroll-loops)
    else()
        add_compile_options(-g -O0)
    endif()
    if(NOCTA_ENABLE_SIMD)
        add_compile_options(-mavx2 -mfma)
        add_compile_definitions(NOCTA_SIMD_ENABLED)
    endif()
endif()

# ============================================
# Sources
# ============================================
set(NOCTA_SOURCES
    src/core/tensor.c
    src/core/memory.c
    src/core/device.c
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
    src/nn/dropout.c
    src/optim/optimizer.c
    src/optim/sgd.c
    src/optim/adam.c
)

# CUDA sources
if(NOCTA_ENABLE_CUDA)
    set(NOCTA_CUDA_SOURCES
        src/cuda/cuda_device.cu
        src/cuda/cuda_elementwise.cu
        src/cuda/cuda_activation.cu
        src/cuda/cuda_matmul.cu
        src/cuda/cuda_reduction.cu
        src/cuda/cuda_conv.cu
        src/cuda/cuda_batchnorm.cu
        src/cuda/cuda_loss.cu
    )
endif()

# ============================================
# Library
# ============================================
if(NOCTA_ENABLE_CUDA)
    add_library(nocta STATIC ${NOCTA_SOURCES} ${NOCTA_CUDA_SOURCES})
    set_target_properties(nocta PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(nocta PUBLIC CUDA::cudart CUDA::cublas)
else()
    add_library(nocta STATIC ${NOCTA_SOURCES})
endif()

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

# Link OpenMP to the library
if(NOCTA_ENABLE_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(nocta PUBLIC OpenMP::OpenMP_C)
endif()

# ============================================
# Examples
# ============================================
if(NOCTA_BUILD_EXAMPLES)
    add_executable(example_xor examples/xor.c)
    target_link_libraries(example_xor PRIVATE nocta)
    
    add_executable(example_save_load examples/save_load.c)
    target_link_libraries(example_save_load PRIVATE nocta)
    
    add_executable(example_mnist examples/mnist.c)
    target_link_libraries(example_mnist PRIVATE nocta)
    
    add_executable(example_batchnorm examples/batchnorm.c)
    target_link_libraries(example_batchnorm PRIVATE nocta)

    add_executable(benchmark examples/benchmark.c)
    target_link_libraries(benchmark PRIVATE nocta)

    add_executable(dropout_test examples/dropout_test.c)
    target_link_libraries(dropout_test PRIVATE nocta)

    add_executable(benchmark_gpu examples/benchmark_gpu.c)
    target_link_libraries(benchmark_gpu PRIVATE nocta)
endif()

# ============================================
# Testing
# ============================================
enable_testing()

add_executable(unit_tests
    tests/test_main.c
    tests/test_tensor.c
    tests/test_ops.c
    tests/test_autograd.c
    tests/test_nn.c
    tests/test_optim.c
    tests/test_io.c
    tests/test_reduction.c
)

target_link_libraries(unit_tests PRIVATE nocta)

add_test(NAME Tensors COMMAND unit_tests tensor)
add_test(NAME Operations COMMAND unit_tests ops)
add_test(NAME Autograd COMMAND unit_tests autograd)
add_test(NAME NeuralNet COMMAND unit_tests nn)
add_test(NAME Optimizers COMMAND unit_tests optim)
add_test(NAME IO COMMAND unit_tests io)
add_test(NAME Reductions COMMAND unit_tests reduction)