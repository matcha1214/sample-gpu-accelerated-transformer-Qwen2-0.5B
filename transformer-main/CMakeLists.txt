cmake_minimum_required(VERSION 3.27)
project(transformer CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)
find_package(CUDAToolkit)


add_compile_options(-Wno-changes-meaning)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)

add_library(transformer_lib
        src/vendor/argparse.hpp
        src/vendor/safetensors.hh
        src/vendor/safetensors_impl.cpp
        src/CudaBuffer.cuh
        src/CudaBuffer.cu
        src/ErrorCheck.h
        src/qwen2/Qwen2Config.h
        src/qwen2/Qwen2Layer.cuh
        src/qwen2/Qwen2Model.cuh
        src/qwen2/Qwen2Loader.h
        src/qwen2/Qwen2Loader.cu
        src/HostBuffer.cpp
        src/HostBuffer.h
        src/HostBuffer.cpp
        src/HostBuffer.h
        src/BPE.cpp
        src/BPE.h
        src/gpu_ops/ArgMax.cu
        src/gpu_ops/ArgMax.cuh
        src/gpu_ops/LayerNorm.cu
        src/gpu_ops/LayerNorm.cuh
        src/gpu_ops/RoPE.cu
        src/gpu_ops/RoPE.cuh
        src/gpu_ops/MatrixVectorMultiply.cu
        src/gpu_ops/MatrixVectorMultiply.cuh
        src/gpu_ops/SiLUMult.cu
        src/gpu_ops/SiLUMult.cuh
        src/gpu_ops/GroupQueryAttention.cuh)

target_link_libraries(transformer_lib PRIVATE CUDA::cudart CUDA::nvtx3)

add_executable(transformer
        src/vendor/argparse.hpp
        src/main.cu
)
target_link_libraries(transformer PRIVATE transformer_lib)

# tests
enable_testing()

add_executable(bpetest ./src/tests/BPETest.cu)
target_link_libraries(bpetest PRIVATE transformer_lib)
add_test(NAME bpetest COMMAND bpetest)

add_executable(argmaxtest ./src/tests/ArgMaxTest.cu)
target_link_libraries(argmaxtest PRIVATE transformer_lib)
add_test(NAME argmaxtest COMMAND argmaxtest)

add_executable(matvecmulttest ./src/tests/MatVecMultTest.cu)
target_link_libraries(matvecmulttest PRIVATE transformer_lib)
add_test(NAME matvecmulttest COMMAND matvecmulttest)

add_executable(layernormtest ./src/tests/LayerNormTest.cu)
target_link_libraries(layernormtest PRIVATE transformer_lib)
add_test(NAME layernormtest COMMAND layernormtest)

add_executable(ropetest ./src/tests/RoPETest.cu)
target_link_libraries(ropetest PRIVATE transformer_lib)
add_test(NAME ropetest COMMAND ropetest)

add_executable(silumulttest ./src/tests/SiLUMultTest.cu)
target_link_libraries(silumulttest PRIVATE transformer_lib)
add_test(NAME silumulttest COMMAND silumulttest)

add_executable(gqatest ./src/tests/GQATest.cu)
target_link_libraries(gqatest PRIVATE transformer_lib)
add_test(NAME gqatest COMMAND gqatest)
