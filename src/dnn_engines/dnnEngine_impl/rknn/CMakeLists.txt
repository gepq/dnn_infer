cmake_minimum_required(VERSION 3.12)

project(rknn_Engine VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

if(DEFINED BSP_LIB_PATH)
    set(RKNN_RT_LIB ${BSP_LIB_PATH}/librknnrt.so)
else()
    set(RKNN_RT_LIB /usr/lib/librknnrt.so)
endif()

set(SOURCES
    rknn.cpp
)


add_library(${PROJECT_NAME} SHARED ${SOURCES})


target_include_directories(${PROJECT_NAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)


string(COMPARE EQUAL ${PROJECT_NAME} ${CMAKE_PROJECT_NAME} is_top_level)
if(is_top_level)
  message(FATAL_ERROR "This subproject must be built as part of the top-level project.")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
target_link_libraries(${PROJECT_NAME} PUBLIC ${RKNN_RT_LIB} common)

install(TARGETS rknn_Engine
    LIBRARY DESTINATION lib
)
