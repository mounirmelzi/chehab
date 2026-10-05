# CKKS module sources
set(FHECO_SOURCE_FILES ${FHECO_SOURCE_FILES}
    ${CMAKE_CURRENT_LIST_DIR}/ckks_params.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ckks_scale_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/level_dag.cpp
    PARENT_SCOPE
)

# Install headers
install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/ckks_params.hpp
        ${CMAKE_CURRENT_LIST_DIR}/ckks_scale_manager.hpp
        ${CMAKE_CURRENT_LIST_DIR}/level_dag.hpp

    DESTINATION
        ${FHECO_INCLUDES_INSTALL_DIR}/ckks
)

