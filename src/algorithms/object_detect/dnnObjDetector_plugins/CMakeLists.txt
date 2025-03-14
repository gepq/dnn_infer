option(ENABLE_YOLOV5 "Enable support for YOLOV5" ON)
option(ENABLE_YOLOV8 "Enable support for YOLOV8" OFF)

if(ENABLE_YOLOV5)
    add_subdirectory(yolov5)
endif()
if(ENABLE_YOLOV8)
    add_subdirectory(yolov8)
endif()
