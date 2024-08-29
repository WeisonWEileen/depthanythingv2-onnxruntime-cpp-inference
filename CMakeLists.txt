cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Examples VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
# find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /home/weison/Downloads/onnxruntime-linux-x64-gpu-1.18.0/include/)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /home/weison/Downloads/onnxruntime-linux-x64-gpu-cuda12-1.18.0/onnxruntime-linux-x64-gpu-1.18.0/include/)

# find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /home/weison/Downloads/onnxruntime-linux-x64-gpu-1.18.0/lib)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /home/weison/Downloads/onnxruntime-linux-x64-gpu-cuda12-1.18.0/onnxruntime-linux-x64-gpu-1.18.0/lib/)

# set(ONNX_RUNTIME_LIB02  /home/weison/Downloads/onnxruntime-linux-x64-gpu-1.18.0/lib)
set(ONNX_RUNTIME_LIB02  /home/weison/Downloads/onnxruntime-linux-x64-gpu-cuda12-1.18.0/onnxruntime-linux-x64-gpu-1.18.0/lib/)
set(ADD_LIB /usr/local/cuda-12.2/targets/x86_64-linux/lib)

add_subdirectory(src)
