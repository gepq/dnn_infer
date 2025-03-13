cmake_minimum_required(VERSION 3.12)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_subdirectory(algorithms)
add_subdirectory(common)
add_subdirectory(dnn_engines)
