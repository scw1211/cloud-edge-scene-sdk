cmake_minimum_required(VERSION 3.10)
# 构建 Jetson 端 RGB PatchCore TensorRT 运行时；同时启用 C++ 和 CUDA。
project(jetson_rgb_trt_runtime_cublas LANGUAGES CXX CUDA)

# TX2/JetPack 4 常见环境仍是 Ubuntu 18.04 + CMake 3.10，因此保持 C++14/CUDA14。
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 依赖 CUDA、OpenCV 图像处理模块，以及 TensorRT 推理库 nvinfer。
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include/aarch64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
find_library(CUBLAS_LIBRARY cublas PATHS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT CUBLAS_LIBRARY)
  message(FATAL_ERROR "TensorRT/cuBLAS headers/libraries were not found. Install JetPack TensorRT/CUDA first.")
endif()

# 主程序由 TensorRT/C++ 控制流和 CUDA PatchCore 后处理两部分组成。
add_executable(rgb_patchcore_trt src/main.cpp src/patchcore_cuda.cu)
target_include_directories(rgb_patchcore_trt PRIVATE
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  src)
target_link_libraries(rgb_patchcore_trt PRIVATE ${TENSORRT_LIBRARY} ${OpenCV_LIBS} ${CUBLAS_LIBRARY} cudart curl yaml-cpp)
# 开启可分离编译，方便 CUDA 源文件和 C++ 主程序分别编译后链接。
set_target_properties(rgb_patchcore_trt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
