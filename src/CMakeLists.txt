# CMakeList.txt : CMake project for PerformanceCF, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_library(LatentFactorModel "LatentFactorModel.cpp" "LatentFactorModel.h")
target_link_libraries(LatentFactorModel Eigen3::Eigen)
# Add source to this project's executable.
add_executable (PerformanceCF "PerformanceCF.cpp" "PerformanceCF.h")
target_link_libraries(PerformanceCF LatentFactorModel)
target_compile_options(PerformanceCF PUBLIC  -std=c++11 -Wall -Wextra)
# TODO: Add tests and install targets if needed.
