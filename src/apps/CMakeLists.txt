add_executable(casm_hella_refactored casm_hella_refactored.cu)
add_executable(casm_hella_orig casm_hella_orig.cu)
add_executable(casm_hella casm_hella.cu)

target_include_directories(
  casm_hella
  PRIVATE
    ${DEDISP_INCLUDE_DIR}
    ${PSRDADA_INCLUDEDIR}
)

target_include_directories(
  casm_hella_orig
  PRIVATE
    ${DEDISP_INCLUDE_DIR}
    ${PSRDADA_INCLUDEDIR}
)

target_include_directories(
  casm_hella_refactored
  PRIVATE
    hella-core
)

target_link_libraries(
  casm_hella_orig
  PRIVATE
    spdlog::spdlog
  PUBLIC
    ${PSRDADA_LDFLAGS}
    ${DEDISP_LIBRARIES}
    CUDA::cudart
    CUDA::nppial
    CUDA::nppif
    CUDA::npps
    CUDA::nppisu
    CUDA::nppc
    CUDA::cublas
)

target_link_libraries(
  casm_hella
  PRIVATE
    spdlog::spdlog
  PUBLIC
    ${PSRDADA_LDFLAGS}
    ${DEDISP_LIBRARIES}
    CUDA::cudart
    CUDA::nppial
    CUDA::nppif
    CUDA::npps
    CUDA::nppisu
    CUDA::nppc
    CUDA::cublas
)

target_link_libraries(
  casm_hella_refactored
  PUBLIC
    hella-core
)

install(
  TARGETS
  casm_hella
  RUNTIME
  DESTINATION
  bin
)
