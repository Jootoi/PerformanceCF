# CMakeList.txt : CMake project for PerformanceCF, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
include_directories("/usr/local/include/eigen3")
# Add source to this project's executable.
add_subdirectory ("src")

# TODO: Add tests and install targets if needed.
