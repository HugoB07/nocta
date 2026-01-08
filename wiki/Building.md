# Building Nocta

## Prerequisites

*   **CMake** (3.10 or higher)
*   **C Compiler** (GCC, Clang, or MSVC)
*   **(Optional) CUDA Toolkit** (for GPU support)

## Building with CMake

1.  Create a build directory:
    ```bash
    mkdir build && cd build
    ```

2.  Configure the project:
    ```bash
    cmake ..
    ```

3.  Build the project (Release mode recommended):
    ```bash
    cmake --build . --config Release
    ```

## Build Options

You can configure the build by passing `-D<OPTION>=<VALUE>` to CMake.

| Option | Default | Description |
|--------|---------|-------------|
| `NOCTA_BUILD_EXAMPLES` | ON | Build example programs |
| `NOCTA_ENABLE_SIMD` | ON | Enable AVX2 SIMD optimizations |
| `NOCTA_ENABLE_OPENMP` | ON | Enable OpenMP multi-threading |
| `NOCTA_ENABLE_CUDA` | OFF | Enable CUDA/cuBLAS GPU acceleration |

## Building with CUDA

To enable GPU acceleration, ensure you have the CUDA Toolkit installed.

```bash
mkdir build && cd build
cmake .. -DNOCTA_ENABLE_CUDA=ON \
         -DCUDAToolkit_ROOT=/usr/local/cuda-12.x/ \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.x/bin/nvcc
cmake --build . --config Release
```
