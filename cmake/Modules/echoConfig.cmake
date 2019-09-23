INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_ECHO echo)

FIND_PATH(
    ECHO_INCLUDE_DIRS
    NAMES echo/api.h
    HINTS $ENV{ECHO_DIR}/include
        ${PC_ECHO_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    ECHO_LIBRARIES
    NAMES gnuradio-echo
    HINTS $ENV{ECHO_DIR}/lib
        ${PC_ECHO_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ECHO DEFAULT_MSG ECHO_LIBRARIES ECHO_INCLUDE_DIRS)
MARK_AS_ADVANCED(ECHO_LIBRARIES ECHO_INCLUDE_DIRS)

