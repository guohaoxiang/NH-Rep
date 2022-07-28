cmake_minimum_required(VERSION 3.0.0)

project(ParametricSample VERSION 0.1.0 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 11)

IF(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
ENDIF()

if(${MSVC})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${CMAKE_SOURCE_DIR}/Bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG  ${CMAKE_SOURCE_DIR}/Bin)
elseif(${CYGWIN})
SET(EXECUTABLE_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/Bin)
elseif(${MINGW})
SET(EXECUTABLE_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/Bin)
elseif(${UNIX})
SET(EXECUTABLE_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/Bin)
endif(${MSVC})

FILE(GLOB_RECURSE project_cpps main.cpp Mesh3D.cpp Helper.cpp Helper.h Mesh3D.h MyObjLoader.h MyCurve.h MySurf.h MyTuple.h)

include_directories(${CMAKE_SOURCE_DIR}/yaml-cpp/include/yaml-cpp ${PROJECT_SOURCE_DIR})

set(CMAKE_DEBUG_POSTFIX "d")
add_executable(ParametricSample ${project_cpps})
target_link_libraries(ParametricSample yaml-cpp)