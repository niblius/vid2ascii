cmake_minimum_required(VERSION 2.8)
project( Vid2ASCII )
find_package( OpenCV REQUIRED )
find_package( OpenCL REQUIRED )
include_directories( ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )

add_executable( Vid2ASCII Vid2ASCII.cpp )
target_link_libraries( Vid2ASCII ${OpenCL_LIBRARIES} ${OpenCV_LIBS})
set (CMAKE_CXX_STANDARD 11)
