function(gpudct_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE gpudct gpudct_flags)
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
  )
  target_compile_definitions(${name} PRIVATE
    GPUDCT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT 600)
endfunction()

gpudct_test(test_transform)
gpudct_test(test_rans)
gpudct_test(test_roundtrip)
gpudct_test(test_robustness)
gpudct_test(test_deblock)
gpudct_test(test_metrics)
gpudct_test(test_bounds)
gpudct_test(test_golden)
gpudct_test(test_conformance)
gpudct_test(test_threading)
gpudct_test(test_cuda)

# Corpus writer. Deliberately not an add_test: see tests/make_conformance.cpp.
add_executable(make_conformance make_conformance.cpp)
target_link_libraries(make_conformance PRIVATE gpudct gpudct_flags)
target_include_directories(make_conformance PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src
)
target_compile_definitions(make_conformance PRIVATE
  GPUDCT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
