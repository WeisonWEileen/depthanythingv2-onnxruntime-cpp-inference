cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
# set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_INCLUDE_DIR})

# add_executable(inference inference.cpp)
add_executable(pred dep_any_v2_onnx.cpp pred_main.cpp)
add_executable(main main.cpp)
# add_executable(inference_deAny dep_any_v2_onnx.cpp)
# target_include_directories(inference PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${SELF_INCLUDE_DIR})
target_include_directories(pred PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${SELF_INCLUDE_DIR})
target_include_directories(main PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${SELF_INCLUDE_DIR})
# target_include_directories(inference_deAny PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${SELF_INCLUDE_DIR})


# target_link_libraries(inference PUBLIC ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
# target_link_directories(inference PUBLIC ${ONNX_RUNTIME_LIB02} ${ADD_LIB})

target_link_libraries(pred PUBLIC ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
target_link_directories(pred PUBLIC ${ONNX_RUNTIME_LIB02} ${ADD_LIB})
target_link_libraries(main PUBLIC ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
target_link_directories(main PUBLIC ${ONNX_RUNTIME_LIB02} ${ADD_LIB})