cmake_minimum_required(VERSION 3.13)

message(STATUS "Building ...")

# Project configuration, specifying version, languages,
# and the C++ standard to use for the whole project
set(PACKAGE_NAME CASM_HELLA)
set(DESCRIPTION "FRB Detection Pipeline for the Coherent All Sky Monitor.")
file(STRINGS version.txt CASM_HELLA_VERSION)

project(${PACKAGE_NAME}
  VERSION ${CASM_HELLA_VERSION}
  DESCRIPTION ${DESCRIPTION}
  LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(CTest)

# External projects
# if (BUILD_TESTING)
#     add_subdirectory(resources/googletest googletest EXCLUDE_FROM_ALL)
# endif()

find_package(CUDAToolkit REQUIRED)
find_package(DEDISP REQUIRED)
find_package(PkgConfig)

pkg_check_modules(SIGPROC sigproc)
pkg_check_modules(PSRDADA REQUIRED psrdada)

add_subdirectory(src)
