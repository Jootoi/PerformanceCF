# CMakeList.txt : CMake project for PerformanceCF, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (OpenMP)
add_library(Utilities "Utilities.h" "Utilities.cpp")
target_link_libraries(Utilities OpenMP::OpenMP_CXX Eigen3::Eigen)
target_compile_options(Utilities PUBLIC -std=c++11 -Wall -Wextra -O3 -march=native -mavx2 -mfma -ffast-math -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized)

add_library(LatentFactorModel "LatentFactorModel.cpp" "LatentFactorModel.h")
target_link_libraries(LatentFactorModel Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_options(LatentFactorModel PUBLIC  -std=c++11 -Wall -Wextra -O3 -fopenmp -march=native -mavx2 -mfma -ffast-math -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized)
# Add source to this project's executable.
add_executable (PerformanceCF "PerformanceCF.cpp" "PerformanceCF.h")
target_link_libraries(PerformanceCF LatentFactorModel)
target_compile_options(PerformanceCF PUBLIC  -std=c++11 -Wall -Wextra)
# TODO: Add tests and install targets if needed.
