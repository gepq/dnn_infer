# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.12)

# Set the project name
project(dnnObjDetector VERSION 0.0.1 LANGUAGES CXX)

# Set the language version
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Add the source files
set(SOURCES
  dnnObjDetector.cpp
)

# Add the library target
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# target_link_libraries(${PROJECT_NAME} PRIVATE bsp_shared)

target_include_directories(${PROJECT_NAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)


string(COMPARE EQUAL ${PROJECT_NAME} ${CMAKE_PROJECT_NAME} is_top_level)
if(is_top_level)
  message(FATAL_ERROR "This subproject must be built as part of the top-level project.")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
target_link_libraries(${PROJECT_NAME} PRIVATE common)

option(ENABLE_RKNN "Enable support for RKNN" ON)
option(ENABLE_TENSORRT "Enable support for TensorRT" OFF)

if(ENABLE_RKNN)
  target_link_libraries(${PROJECT_NAME} PUBLIC rknn_Engine)
endif()


install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
)


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION include/${CMAKE_PROJECT_NAME}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
)

# add_subdirectory(dnnObjDetector_plugins)
