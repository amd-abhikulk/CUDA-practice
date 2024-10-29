
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)
set(CMAKE_CUDA_TOOLKIT_ROOT_DIR  /usr/local/cuda-12.4)
project(MyProject LANGUAGES CXX CUDA)

set (TORCH_PATH "/home/arunGPU12/.local/lib/python3.10/site-packages/torch/share/cmake")
# Set the path to the LibTorch installation
set(CMAKE_PREFIX_PATH ${TORCH_PATH})

set(CUDA_LIBRARIES  /usr/local/cuda-12.4/lib64)
set(CMAKE_INCLUDE_PATH  /usr/local/cuda-12.4/include)
#find_package(CUDA REQUIRED PATHS ${CMAKE_CUDA_TOOLKIT_ROOT_DIR})
find_package(Torch REQUIRED)

add_executable(my_project main.cu)
target_link_libraries(my_project "${TORCH_LIBRARIES}" ${CUDA_LIBRARIES})

set_property(TARGET my_project PROPERTY CXX_STANDARD 17)
