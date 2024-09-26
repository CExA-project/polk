add_library(
    polk
    INTERFACE
)

target_include_directories(
    polk
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<INSTALL_INTERFACE:.>
)

target_link_libraries(
    polk
    INTERFACE
        Kokkos::kokkos
)

install(
    DIRECTORY
        "${CMAKE_CURRENT_LIST_DIR}/polk"
    DESTINATION
        include
)

add_library(
    Polk::polk
    ALIAS
        polk
)
