add_executable(gpudct_cli
  gpudct/main.cpp
  gpudct/raw_io.cpp
)
set_target_properties(gpudct_cli PROPERTIES OUTPUT_NAME gpudct)
target_link_libraries(gpudct_cli PRIVATE gpudct gpudct_flags)
target_include_directories(gpudct_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpudct ${CMAKE_SOURCE_DIR}/src)

add_executable(train_tables train_tables/main.cpp)
target_link_libraries(train_tables PRIVATE gpudct gpudct_flags)
target_include_directories(train_tables PRIVATE ${CMAKE_SOURCE_DIR}/src)
