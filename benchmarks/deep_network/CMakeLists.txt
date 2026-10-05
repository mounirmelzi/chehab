add_executable(deep_network deep_network.cpp)
target_link_libraries(deep_network fheco)

add_executable(test_orion_comparison test_orion_comparison.cpp)
target_link_libraries(test_orion_comparison fheco)

add_executable(benchmark_detailed benchmark_detailed.cpp)
target_link_libraries(benchmark_detailed fheco)
