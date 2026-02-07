cmake_minimum_required(VERSION 3.13)

project(HEConv2D VERSION 1.0 LANGUAGES CXX)
find_package(SEAL 4.1 REQUIRED)

add_library(gen_he_conv2d STATIC _gen_he_conv2d.cpp)
target_compile_features(gen_he_conv2d PUBLIC cxx_std_17)
set_target_properties(gen_he_conv2d PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(gen_he_conv2d SEAL::seal)

add_library(utils STATIC utils.cpp)
target_compile_features(utils PUBLIC cxx_std_17)
set_target_properties(utils PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(utils SEAL::seal)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE gen_he_conv2d)
target_link_libraries(main PRIVATE utils)

