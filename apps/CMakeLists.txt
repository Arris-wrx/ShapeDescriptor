add_executable(demo demo.cpp)
target_link_libraries(demo origami)


set_target_properties(demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
