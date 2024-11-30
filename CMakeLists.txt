cmake_minimum_required(VERSION 3.10)

# Project name
project(NNet)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source directory
include_directories(cpp)

# Add the executable
add_executable(NNet
    cpp/main.cpp
    cpp/nnet.cpp
)
