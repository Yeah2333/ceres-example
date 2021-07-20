cmake_minimum_required(VERSION 3.16)
project(ceres_example)

set(CMAKE_CXX_STANDARD 14)

#find_package(Glog REQUIRED)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(helloword src/helloword.cpp)
target_link_libraries(helloword ${CERES_LIBRARIES})

add_executable(powell src/powell.cpp)
target_link_libraries(powell ${CERES_LIBRARIES})