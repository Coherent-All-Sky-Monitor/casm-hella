add_executable(casm_hella_refactored casm_hella_refactored.cu)

target_include_directories(
  casm_hella_refactored
  PRIVATE
    hella-core
)

target_link_libraries(
  casm_hella_refactored
  PUBLIC
    hella-core
)

install(
  TARGETS
  casm_hella_refactored
  RUNTIME
  DESTINATION
  bin
)
