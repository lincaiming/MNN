cmake_minimum_required(VERSION 3.0)
project(yolov8_demo)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# # include dir
# include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# # libs dir
# link_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../include/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../tools/cv/include/)

link_directories(${CMAKE_CURRENT_LIST_DIR}/../../build)
link_directories(${CMAKE_CURRENT_LIST_DIR}/../../build/tools/cv)
link_directories(${CMAKE_CURRENT_LIST_DIR}/../../build/express)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



# source files
FILE(GLOB SRCS  ${CMAKE_CURRENT_LIST_DIR}/yolov8_demo.cpp)

# target
add_executable(yolov8_demo ${SRCS})
add_executable(yolov5 ${CMAKE_CURRENT_LIST_DIR}/yolov5.cpp ${CMAKE_CURRENT_LIST_DIR}/util.cpp)
target_link_libraries(yolov5 MNN MNN_Express MNNOpenCV ${OpenCV_LIBS})

# link
if (MSVC)
    target_link_libraries(yolov8_demo MNN)
else()
    target_link_libraries(yolov8_demo MNN MNN_Express MNNOpenCV
                            ${OpenCV_LIBS})
endif()
