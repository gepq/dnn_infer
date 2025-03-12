cmake_minimum_required(VERSION 3.12)

project(common VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")


if(DEFINED BSP_PKG_CONFIG_PATH)
    set(ENV{PKG_CONFIG_PATH} ${BSP_PKG_CONFIG_PATH})
    message(STATUS "ENV{PKG_CONFIG_PATH}: $ENV{PKG_CONFIG_PATH}")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(PC_SPDLOG REQUIRED spdlog)
pkg_check_modules(PC_CLI11 REQUIRED CLI11)


set(SOURCES
  ArgParser.hpp
  Logger.hpp
  Logger.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

if(PC_SPDLOG_LIBRARIES)
  target_link_options(${PROJECT_NAME} PUBLIC ${PC_SPDLOG_LDFLAGS} "-Wl,-rpath,${PC_SPDLOG_LIBRARY_DIRS}")
  # target_link_libraries(${PROJECT_NAME} PUBLIC ${PC_SPDLOG_LIBRARIES})
  target_include_directories(${PROJECT_NAME} PUBLIC ${PC_SPDLOG_INCLUDE_DIRS})
endif()

if(PC_CLI11_CFLAGS)
  target_include_directories(${PROJECT_NAME} PUBLIC ${PC_CLI11_INCLUDE_DIRS})
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

string(COMPARE EQUAL ${PROJECT_NAME} ${CMAKE_PROJECT_NAME} is_top_level)
if(is_top_level)
  message(FATAL_ERROR "This subproject must be built as part of the top-level project.")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

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
