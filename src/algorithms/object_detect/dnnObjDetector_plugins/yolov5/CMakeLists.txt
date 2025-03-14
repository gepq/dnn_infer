cmake_minimum_required(VERSION 3.12)

project(YOLOV5Plugin VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

if(DEFINED BSP_PKG_CONFIG_PATH)
    set(ENV{PKG_CONFIG_PATH} ${BSP_PKG_CONFIG_PATH})
    message(STATUS "ENV{PKG_CONFIG_PATH}: $ENV{PKG_CONFIG_PATH}")
endif()

if(DEFINED BSP_LIB_PATH)
    set(OpenCV_DIR ${BSP_LIB_PATH}/cmake/opencv4)
else()
    set(OpenCV_DIR /usr/lib/cmake/opencv4)
endif()

find_package(OpenCV REQUIRED)

# include_directories(${CMAKE_SOURCE_DIR}/src/algorithms/object_detect)

# Create the shared library
set(PLUGIN_NAME yolov5)

add_library(${PLUGIN_NAME} SHARED
    yolov5.cpp
)

target_include_directories(${PLUGIN_NAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

if(OpenCV_LIBRARIES)
    target_link_options(${PLUGIN_NAME} PUBLIC "-Wl,-rpath,${OpenCV_LIBRARY_DIRS}" ${OpenCV_INCLUDE_LDFLAGS} )
    target_include_directories(${PLUGIN_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()
set_target_properties(${PLUGIN_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

install(TARGETS ${PLUGIN_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
