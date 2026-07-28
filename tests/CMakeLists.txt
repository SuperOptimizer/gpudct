function(gpudct_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE gpudct gpudct_flags)
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
  )
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT 600)
endfunction()

gpudct_test(test_transform)
gpudct_test(test_rans)
gpudct_test(test_roundtrip)
gpudct_test(test_robustness)
gpudct_test(test_deblock)
gpudct_test(test_bounds)
gpudct_test(test_golden)
gpudct_test(test_cuda)
