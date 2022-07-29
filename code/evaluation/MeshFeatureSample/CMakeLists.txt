cmake_minimum_required(VERSION 3.10)

# Set the project name
project (SimpleSample)

# Create a sources variable with a link to all cpp files to compile
set(SOURCES
    SimpleSample.cpp
    Mesh3D.cpp
)

#SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${CMAKE_SOURCE_DIR}/Bin)
#SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG  ${CMAKE_SOURCE_DIR}/Bin)

# Add an executable with the above sources
add_executable(SimpleSample ${SOURCES})

# Set the directories that should be included in the build command for this target
# when running g++ these will be included as -I/directory/path/
target_include_directories(SimpleSample
    PRIVATE 
        ${CMAKE_SOURCE_DIR}
)