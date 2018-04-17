cmake_minimum_required(VERSION 3.6)
project(MTCNN)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS} "/home/dev/caffe/distribute/include")

# minicaffe
#include_directories(/home/yzh/working/mini-caffe/include)
#link_directories(/home/yzh/working/mini-caffe/build)

set(SOURCE_FILES main.cpp MTCNN.cpp MTCNN.h posface.cpp posface.h)
add_executable(MTCNN ${SOURCE_FILES})

target_link_libraries(MTCNN ${OpenCV_LIBS} )
target_link_libraries(MTCNN ${Caffe_LIBRARIES})

