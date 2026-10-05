cmake_minimum_required(VERSION 3.16)

add_executable(conv2d conv2d.cpp)
target_link_libraries(conv2d PRIVATE fheco)
target_compile_features(conv2d PRIVATE cxx_std_17)

# Optimized Conv2D benchmark
add_executable(conv2d_optimized_benchmark conv2d_optimized_benchmark.cpp)
target_link_libraries(conv2d_optimized_benchmark PRIVATE fheco)
target_compile_features(conv2d_optimized_benchmark PRIVATE cxx_std_17)

# Create he subdirectory for SEAL output
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/he)

# Copy helper files
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/fhe_io_example.txt
    ${CMAKE_CURRENT_BINARY_DIR}/fhe_io_example.txt
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/he/CMakeLists.txt
    ${CMAKE_CURRENT_BINARY_DIR}/he/CMakeLists.txt
    COPYONLY
)

