cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)


find_package(OpenCV REQUIRED)

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x")

file(GLOB hdrs include/*.h*)
file(GLOB srcs src/*.cpp src/*.h*)

include_directories(include src)
add_library(origami STATIC ${srcs} ${hdrs} README.md)
target_link_libraries(origami ${OpenCV_LIBS})


add_subdirectory(apps)


