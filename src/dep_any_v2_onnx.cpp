/*
 * Author: weison pan
 * Date: 8,2024
 * Email: weisonweileen@gmail.com
 */
#include <cstring>
#include <iostream>
#include <vector>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "onnxruntime_cxx_api.h"

#include "include/dep_ang_v2_onnx.hpp"

DepthAnyV2Onnx::DepthAnyV2Onnx(const char* onnx_model_path): onnx_model_path_(onnx_model_path)
{
    // this->printInputOutputInfo();
}

cv::Mat DepthAnyV2Onnx::predict(const cv::Mat& img)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DepthAnyV2Onnx");
    Ort::Session session(env, this->onnx_model_path_, Ort::SessionOptions());
    printInputOutputInfo(session);
    auto memory_info_handler =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    cv::Mat preprocessed_img = this->preprocessImg(img);

    std::vector<cv::Mat> mat_channels;
    std::cout << "The final cv::Mat input type is " << preprocessed_img.type()
              << std::endl;

    // just take one batch
    std::vector<float> tensor_values_handler(product(this->input_node_dims_));
    std::copy(preprocessed_img.begin<float>(), preprocessed_img.end<float>(),
              tensor_values_handler.begin());

    std::cout << "input tensor value vector size is "
              << tensor_values_handler.size() << std::endl;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_handler, tensor_values_handler.data(),
        product(this->input_node_dims_), this->input_node_dims_.data(),
        this->input_node_dims_.size());
    assert(input_tensor.IsTensor());

    std::vector<float> output_tensor_values;
    std::vector<Ort::Value> output_tensors;
    output_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, output_tensor_values.data(), product(this->output_node_dims_),
        this->output_node_dims_.data(), this->output_node_dims_.size()));

    
    session.Run(Ort::RunOptions{nullptr}, this->input_node_names_.data(),
                 &input_tensor, 1, this->output_node_names_.data(),
                 output_tensors.data(), 1);
    std::vector<float> output_data;

    
    
    cv::Mat output_tensor(output_data);

    // output_tensor =
    //     output_tensor.reshape(1, {300,1000});

    double min_val = 0.0, max_val = 0.0;
    cv::minMaxLoc(output_tensor, &min_val, &max_val);
    output_tensor.convertTo(output_tensor, CV_32F);
    output_tensor = (output_tensor - min_val) / (max_val - min_val) *= 255.0f;
    cv::applyColorMap(output_tensor, output_tensor, cv::COLORMAP_JET);

    return output_tensor;
}

/**
@brief return 1×3×518×518，FP32 Ort::Value for vitb.onnx input
 *
 * @param mat
 * @return Ort::Value
 */
cv::Mat
DepthAnyV2Onnx::preprocessImg(const cv::Mat& mat)
{
    cv::Mat resize_mat, resize_rgb_mat, scaled_mat, blob_mat;
    cv::resize(mat, resize_mat, cv::Size(518, 518), cv::INTER_LINEAR);
    cv::cvtColor(resize_mat, resize_rgb_mat, cv::COLOR_BGR2RGB);
    resize_rgb_mat.convertTo(scaled_mat, CV_32F, 1.0 / 255);
    cv::dnn::blobFromImage(scaled_mat, blob_mat);

    return blob_mat;
}

/**
 * @brief get and print onnx model input and output information
 * @param void
 * API reference
 * :https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
 */
void DepthAnyV2Onnx::printInputOutputInfo(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    // Ort::AllocatedStringPtr input_name_ptr =
        // session_.GetInputNameAllocated(0, allocator);
    // const char* input_name = input_name_ptr.get();
    // this->input_node_names_.push_back(input_name);

    // auto a = this->input_node_names_.data();
    
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto inpu_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    this->input_node_dims_ = inpu_tensor_info.GetShape();
    ONNXTensorElementDataType input_type = inpu_tensor_info.GetElementType();

    // Ort::AllocatedStringPtr output_name_ptr =
        // session_.GetOutputNameAllocated(0, allocator);
    // const char* output_name = output_name_ptr.get();
    // this->output_node_names_.push_back(output_name);
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType output_type = output_tensor_info.GetElementType();
    this->output_node_dims_ = output_tensor_info.GetShape();

    std::cout << "Input nodes number " << num_input_nodes << std::endl;
    std::cout << "Input Name: " << *(this->input_node_names_.data()) << std::endl;
    std::cout << "Input Type: " << input_type << std::endl;
    std::cout << "Input Dimensions: ";
    for (const auto& dim : this->input_node_dims_)
    {
        std::cout << dim << ' ';
    }
    std::cout << std::endl;

    std::cout << "Output nodes number " << num_output_nodes << std::endl;
    std::cout << "Output Name: " <<*(this->output_node_names_.data()) << std::endl;
    std::cout << "Output Type: " << output_type << std::endl;
    std::cout << "Output Dimensions: ";
    for (const auto& dim : this->output_node_dims_)
    {
        std::cout << dim << ' ';
    }
    std::cout << std::endl;
}