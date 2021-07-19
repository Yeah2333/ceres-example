cmake_minimum_required(VERSION 3.19)
project(ceres_example)

set(CMAKE_CXX_STANDARD 14)

find_package(ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(ceres_example main.cpp)
target_link_libraries(ceres_example ${CERES_LIBS})