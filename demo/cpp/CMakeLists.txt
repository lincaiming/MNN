cmake_minimum_required(VERSION 3.0)
project(yolov8_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# # include dir
# include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# # libs dir
# link_directories(${CMAKE_CURRENT_LIST_DIR}/lib)


include_directories(${CMAKE_CURRENT_LIST_DIR}/../../include/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../tools/cv/include/)

link_directories(${CMAKE_CURRENT_LIST_DIR}/../../build)
link_directories(${CMAKE_CURRENT_LIST_DIR}/../../build/tools/cv)
link_directories(${CMAKE_CURRENT_LIST_DIR}/../../build/express)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/yolov8_demo.cpp)

# target
add_executable(yolov8_demo ${SRCS})

# link
if (MSVC)
    target_link_libraries(yolov8_demo MNN)
else()
    target_link_libraries(yolov8_demo MNN MNN_Express MNNOpenCV)
endif()
