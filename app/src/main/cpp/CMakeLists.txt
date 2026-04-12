cmake_minimum_required(VERSION 3.22.1)

project("tunedllm")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(LLAMA_DIR /data/data/com.termux/files/home/llama.cpp)
set(OPENBLAS_LIB /data/data/com.termux/files/usr/lib/libopenblas.so)

# Include directories
include_directories(
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/src
    ${LLAMA_DIR}/src/models
    ${LLAMA_DIR}/ggml/include
    ${LLAMA_DIR}/ggml/src
    ${LLAMA_DIR}/ggml/src/ggml-cpu
    ${LLAMA_DIR}/ggml/src/ggml-cpu/arch/arm
)

# GGML core sources
set(GGML_SOURCES
    ${LLAMA_DIR}/ggml/src/ggml.c
    ${LLAMA_DIR}/ggml/src/ggml.cpp
    ${LLAMA_DIR}/ggml/src/ggml-alloc.c
    ${LLAMA_DIR}/ggml/src/ggml-backend.cpp
    ${LLAMA_DIR}/ggml/src/ggml-backend-dl.cpp
    ${LLAMA_DIR}/ggml/src/ggml-backend-reg.cpp
    ${LLAMA_DIR}/ggml/src/ggml-backend-meta.cpp
    ${LLAMA_DIR}/ggml/src/ggml-opt.cpp
    ${LLAMA_DIR}/ggml/src/ggml-quants.c
    ${LLAMA_DIR}/ggml/src/ggml-threading.cpp
    ${LLAMA_DIR}/ggml/src/gguf.cpp
    
    # CPU Backend
    ${LLAMA_DIR}/ggml/src/ggml-cpu/ggml-cpu.c
    ${LLAMA_DIR}/ggml/src/ggml-cpu/ggml-cpu.cpp
    ${LLAMA_DIR}/ggml/src/ggml-cpu/binary-ops.cpp
    ${LLAMA_DIR}/ggml/src/ggml-cpu/ops.cpp
    ${LLAMA_DIR}/ggml/src/ggml-cpu/quants.c
    ${LLAMA_DIR}/ggml/src/ggml-cpu/repack.cpp
    ${LLAMA_DIR}/ggml/src/ggml-cpu/traits.cpp
    ${LLAMA_DIR}/ggml/src/ggml-cpu/unary-ops.cpp
    ${LLAMA_DIR}/ggml/src/ggml-cpu/vec.cpp
    
    # ARM specific backend
    ${LLAMA_DIR}/ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp
    ${LLAMA_DIR}/ggml/src/ggml-cpu/arch/arm/quants.c
    ${LLAMA_DIR}/ggml/src/ggml-cpu/arch/arm/repack.cpp
)

# Llama sources from src/
file(GLOB LLAMA_SOURCES "${LLAMA_DIR}/src/*.cpp")

# Llama model implementations from src/models/
file(GLOB MODEL_SOURCES "${LLAMA_DIR}/src/models/*.cpp")

add_library(llama_jni SHARED
    jni_bridge.cpp
    ${GGML_SOURCES}
    ${LLAMA_SOURCES}
    ${MODEL_SOURCES}
)

find_package(OpenMP REQUIRED)

target_link_libraries(llama_jni
    ${OPENBLAS_LIB}
    OpenMP::OpenMP_CXX
    log
    android
    dl
)

target_compile_options(llama_jni PRIVATE
    -O3
    -fopenmp
    -DGGML_USE_OPENBLAS
    -DGGML_USE_CPU
    -DGGML_VERSION="0.9.11"
    -DGGML_COMMIT="69c28f154"
)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_compile_options(llama_jni PRIVATE -DGGML_CPU_ARM_ARCH=8)
endif()
