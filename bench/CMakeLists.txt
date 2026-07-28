add_executable(bench_rd bench_rd.cpp)
target_link_libraries(bench_rd PRIVATE gpudct gpudct_flags)
target_include_directories(bench_rd PRIVATE
  ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/src
)
