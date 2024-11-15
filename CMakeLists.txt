cmake_minimum_required(VERSION 3.10)

project(NNet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES
    main.cpp
    nnet.cpp
)

set(HEADERS
    nnet.h
)

add_executable(test_cpp ${SOURCES} tests/test_cpp.cpp)
