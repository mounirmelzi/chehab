cmake_minimum_required(VERSION 3.18)
project(gpu_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(DEFINED ENV{HEONGPU_DIR})
    set(HEONGPU_DIR $ENV{HEONGPU_DIR})
elseif(DEFINED ENV{HOME})
    set(HEONGPU_DIR "$ENV{HOME}/local/heongpu")
endif()

set(NTL_DIR "$ENV{HOME}/local/ntl")
set(GMP_DIR "/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gmp-6.3.0-zims4vx7m6ggtn3ava2r2ksidghaly5v")

message(STATUS "Using HEonGPU from: ${HEONGPU_DIR}")

list(APPEND CMAKE_PREFIX_PATH "${HEONGPU_DIR}")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(rmm QUIET CONFIG)
if(rmm_FOUND)
    message(STATUS "Found RMM via cmake config -- CCCL ABI will match")
else()
    message(STATUS "RMM cmake config not found -- falling back to manual linking")
endif()

# Shared include/link setup for both targets
set(HEONGPU_INCLUDES
    "${HEONGPU_DIR}/include"
    "${HEONGPU_DIR}/include/HEonGPU-1.1"
    "${HEONGPU_DIR}/include/GPUFFT-1.0"
    "${HEONGPU_DIR}/include/GPUNTT-1.0"
    "${HEONGPU_DIR}/include/RNGonGPU-1.0"
    "${NTL_DIR}/include"
    "${GMP_DIR}/include"
)

set(HEONGPU_LIBS
    "${HEONGPU_DIR}/lib/libheongpu.a"
    "${HEONGPU_DIR}/lib64/libfft-1.0.a"
    "${HEONGPU_DIR}/lib64/libntt-1.0.a"
    "${HEONGPU_DIR}/lib64/librngongpu-1.0.a"
)

set(COMMON_LIBS
    "${NTL_DIR}/lib/libntl.so"
    "${GMP_DIR}/lib/libgmp.so"
    CUDA::cudart
    Threads::Threads
    OpenSSL::Crypto
    OpenSSL::SSL
    ZLIB::ZLIB
)

# Helper function to configure a HEonGPU benchmark target
function(add_heongpu_benchmark TARGET SOURCE)
    add_executable(${TARGET} ${SOURCE})
    target_include_directories(${TARGET} BEFORE PRIVATE "${HEONGPU_DIR}/include")
    target_include_directories(${TARGET} PRIVATE ${HEONGPU_INCLUDES})
    target_link_libraries(${TARGET} PRIVATE ${HEONGPU_LIBS})
    if(rmm_FOUND)
        target_link_libraries(${TARGET} PRIVATE rmm::rmm)
    else()
        target_link_libraries(${TARGET} PRIVATE
            "${HEONGPU_DIR}/lib64/librmm.so"
            "${HEONGPU_DIR}/lib64/librapids_logger.so"
        )
    endif()
    target_link_libraries(${TARGET} PRIVATE ${COMMON_LIBS})
    target_compile_definitions(${TARGET} PRIVATE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
    set_target_properties(${TARGET} PROPERTIES
        CUDA_ARCHITECTURES "70"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

add_heongpu_benchmark(gpu_benchmark gpu_benchmark.cu)
add_heongpu_benchmark(gpu_dnn_benchmark gpu_dnn_benchmark.cu)
