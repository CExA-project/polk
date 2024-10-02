include(GNUInstallDirs)

install(
    TARGETS
        polk
    EXPORT
        PolkTargets
    ARCHIVE DESTINATION
        "${CMAKE_INSTALL_LIBDIR}"
)

install(
    EXPORT
        PolkTargets
    NAMESPACE Polk::
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/Polk"
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/PolkConfigVersion.cmake
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/PolkConfigVersion.cmake"
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/Polk"
)
