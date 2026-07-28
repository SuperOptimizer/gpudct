add_library(gpudct
  core/types.cpp
  core/format.cpp
  core/rans.cpp
  core/models.cpp
  core/chunk_codec.cpp
  core/correction.cpp
  core/codec.cpp
  metrics/metrics.cpp
)

target_include_directories(gpudct
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(gpudct PRIVATE gpudct_flags)

if(GPUDCT_ENABLE_CUDA)
  target_sources(gpudct PRIVATE cuda/backend.cu)
  target_compile_definitions(gpudct PRIVATE GPUDCT_HAVE_CUDA)
  set_target_properties(gpudct PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(GPUDCT_ENABLE_SIMD)
  target_compile_definitions(gpudct PRIVATE GPUDCT_HAVE_SIMD)
endif()

find_package(Threads REQUIRED)
target_link_libraries(gpudct PUBLIC Threads::Threads)

add_library(gpudct::gpudct ALIAS gpudct)
