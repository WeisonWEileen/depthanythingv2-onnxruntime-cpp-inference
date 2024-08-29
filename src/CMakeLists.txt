cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(inference inference.cpp)
add_executable(inference_deAny inference.cpp)
target_include_directories(inference PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_include_directories(inference_deAny PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})


target_link_libraries(inference PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
target_link_directories(inference PUBLIC ${ONNX_RUNTIME_LIB02} ${ADD_LIB})

target_link_libraries(inference_deAny PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
target_link_directories(inference_deAny PUBLIC ${ONNX_RUNTIME_LIB02} ${ADD_LIB})