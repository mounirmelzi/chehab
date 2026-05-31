set(UFHE_SOURCE_FILES "")

option(UFHE_USE_HEONGPU "Enable HEonGPU backend (requires CUDA and HEonGPU library)" OFF)

add_subdirectory(src/ufhe/seal_backend)
add_subdirectory(src/ufhe/heongpu_backend)
add_subdirectory(src/ufhe)
add_subdirectory(src/ufhe/api)

add_library(ufhe STATIC ${UFHE_SOURCE_FILES})

target_include_directories(ufhe PUBLIC src)

target_link_libraries(ufhe PUBLIC SEAL::seal)

if(UFHE_USE_HEONGPU)
  target_compile_definitions(ufhe PUBLIC UFHE_USE_HEONGPU)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(ufhe PUBLIC heongpu CUDA::cudart)
  install(FILES ${UFHE_HEONGPU_HEADER_FILES} DESTINATION include/ufhe/heongpu_backend)
endif()

install(FILES ${UFHE_HEADER_FILES} DESTINATION include/ufhe)
install(FILES ${UFHEAPI_HEADER_FILES} DESTINATION include/ufhe/api)
install(FILES ${UFHESEALBACCKED_HEADER_FILES} DESTINATION include/ufhe/seal_backend)
install(TARGETS ufhe LIBRARY DESTINATION lib)

