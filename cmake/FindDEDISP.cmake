# Find the DEDISP library
#
# The following variables are optionally searched for defaults
#  DEDISP_ROOT_DIR: Base directory where all dedisp components are found

# The following are set after configuration is done:
#  DEDISP_FOUND
#  DEDISP_INCLUDE_DIR
#  DEDISP_LIBRARIES
#  DEDISP_LDFLAGS

find_path(DEDISP_INCLUDE_DIR
  NAMES dedisp.h
  HINTS
  ENV INST
  ENV DEDISP_PATH
  ENV DEDISP_ROOT_DIR
  PATH_SUFFIXES include dedisp/include)

find_library(DEDISP_LIBRARIES
  NAMES dedisp
  HINTS
  ENV INST
  ENV DEDISP_ROOT_DIR
  ENV DEDISP_PATH
  PATH_SUFFIXES lib lib64 dedisp/lib dedisp/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DEDISP DEFAULT_MSG DEDISP_INCLUDE_DIR DEDISP_LIBRARIES)

get_filename_component(_DEDISP_LIB_DIR ${DEDISP_LIBRARIES} DIRECTORY)
get_filename_component(_DEDISP_LIB_NAME ${DEDISP_LIBRARIES} NAME)

string(REPLACE "lib" "" _DEDISP_LIB_NAME ${_DEDISP_LIB_NAME})
string(REPLACE ".so" "" _DEDISP_LIB_NAME ${_DEDISP_LIB_NAME})

set(DEDISP_LDFLAGS "-L${_DEDISP_LIB_DIR} -l${_DEDISP_LIB_NAME}")

mark_as_advanced(DEDISP_INCLUDE_DIR DEDISP_LIBRARIES DEDISP_LDFLAGS)

