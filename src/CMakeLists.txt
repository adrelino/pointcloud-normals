# define new project
PROJECT(pointcloud-normals)
CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 11)


include_directories(../include/)

ADD_EXECUTABLE(pointcloud-normals main.cpp)