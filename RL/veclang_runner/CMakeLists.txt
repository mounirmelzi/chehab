file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/RL/veclang_runner/he)
add_executable(veclang_runner veclang_runner.cpp)
target_link_libraries(veclang_runner PRIVATE fheco)

# Define the source file and destination file

set(SOURCE_FILE_2 ${CMAKE_SOURCE_DIR}/benchmarks/utils.cpp)
set(DESTINATION_FILE_2 ${CMAKE_SOURCE_DIR}/build/RL/veclang_runner/he/utils.cpp)

set(SOURCE_FILE_3 ${CMAKE_SOURCE_DIR}/benchmarks/utils.hpp)
set(DESTINATION_FILE_3 ${CMAKE_SOURCE_DIR}/build/RL/veclang_runner/he/utils.hpp)

set(SOURCE_FILE_4 ${CMAKE_SOURCE_DIR}/RL/veclang_runner/he/CMakeLists.txt)
set(DESTINATION_FILE_4 ${CMAKE_SOURCE_DIR}/build/RL/veclang_runner/he/CMakeLists.txt)

set(SOURCE_FILE_5 ${CMAKE_SOURCE_DIR}/RL/veclang_runner/temp/fhe_io_example.txt)
set(DESTINATION_FILE_5 ${CMAKE_SOURCE_DIR}/build/RL/veclang_runner/fhe_io_example.txt)

set(SOURCE_FILE_6 ${CMAKE_SOURCE_DIR}/RL/veclang_runner/temp/inputs.txt)
set(DESTINATION_FILE_6 ${CMAKE_SOURCE_DIR}/build/RL/inputs.txt)

set(SOURCE_FILE_7 ${CMAKE_SOURCE_DIR}/RL/veclang_runner/temp/vectorized_code.txt)
set(DESTINATION_FILE_7 ${CMAKE_SOURCE_DIR}/build/RL/vectorized_code.txt)


add_custom_target(copy_my_file_2veclang_runner ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_2} ${DESTINATION_FILE_2}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_2} to ${DESTINATION_FILE_2}"
)
add_custom_target(copy_my_file_3veclang_runner ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_3} ${DESTINATION_FILE_3}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_3} to ${DESTINATION_FILE_3}"
)
add_custom_target(copy_my_file_4veclang_runner ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_4} ${DESTINATION_FILE_4}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_4} to ${DESTINATION_FILE_4}"
)
add_custom_target(copy_my_file_5veclang_runner ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_5} ${DESTINATION_FILE_5}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_5} to ${DESTINATION_FILE_5}"
)
add_custom_target(copy_my_file_6veclang_runner ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_6} ${DESTINATION_FILE_6}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_6} to ${DESTINATION_FILE_6}"
)
add_custom_target(copy_my_file_7veclang_runner ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_7} ${DESTINATION_FILE_7}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_7} to ${DESTINATION_FILE_7}"
)
