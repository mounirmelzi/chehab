set(FHECO_SOURCE_FILES ${FHECO_SOURCE_FILES}
    ${CMAKE_CURRENT_LIST_DIR}/gen_func.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gen_func_lattigo.cpp
)

install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/constants.hpp
        ${CMAKE_CURRENT_LIST_DIR}/constants_lattigo.hpp
        ${CMAKE_CURRENT_LIST_DIR}/gen_func.hpp
        ${CMAKE_CURRENT_LIST_DIR}/gen_func_lattigo.hpp

    DESTINATION
        ${FHECO_INCLUDES_INSTALL_DIR}/code_gen
)

set(FHECO_SOURCE_FILES ${FHECO_SOURCE_FILES} PARENT_SCOPE)
