#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#define IMG_HEIGHT 1216
#define IMG_WIDTH 352

cv::Mat preprocess_img(const cv::Mat& mat)
{
    cv::Mat resize_mat, resize_rgb_mat, scaled_mat, blob_mat;
    cv::resize(mat, resize_mat, cv::Size(518, 518), cv::INTER_LINEAR);
    cv::cvtColor(resize_mat, resize_rgb_mat, cv::COLOR_BGR2RGB);
    resize_rgb_mat.convertTo(scaled_mat, CV_32F, 1.0 / 255);
    cv::dnn::blobFromImage(scaled_mat, blob_mat);

    return blob_mat;
}

int main(int argc, char* argv[])
{
    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this
    // line to use CUDA for this session (we also need to include
    // cuda_provider_factory.h above which defines it) #include
    // "cuda_provider_factory.h"
    // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    //*************************************************************************
    // create session and load model into memory
    // using squeezenet version 1.3
    // URL = https://github.com/onnx/models/tree/master/squeezenet

    const char* model_path =
        "/home/weison/Desktop/ONNX-Runtime-Inference/data/models/vitb.onnx";

    printf("Using Onnxruntime C++ API\n");
    Ort::Session session(env, model_path, session_options);

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t>
        input_node_dims; // simplify... this model has only 1 input node {1, 3,
                         // 224, 224}. Otherwise need vector<vector<>>

    printf("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    const char* input_name = "l_x_";
    for (int i = 0; i < num_input_nodes; i++)
    {
        // print input node names
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++){
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
        }
    }

    //*************************************************************************
    // Similar operations to get output node information.
    // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
    // OrtSessionGetOutputTypeInfo() as shown above.

    //*************************************************************************
    // Score the model using sample data, and inspect values

    size_t input_tensor_size =
        518 * 518 *
        3; // simplify ... using known dim values to calculate size
           // use OrtGetTensorShapeElementCount() to get official size!

    std::vector<float> input_tensor_values(input_tensor_size);

    cv::Mat img =
        cv::imread("/home/weison/Desktop/ONNX-Runtime-Inference/data/images/"
                   "2011_10_03_drive_0047_sync_image_0000000791_image_03.png");

    cv::Mat preprocessed_img = preprocess_img(img);
    std::copy(preprocessed_img.begin<float>(), preprocessed_img.end<float>(),
              input_tensor_values.begin());

    std::vector<const char*> output_node_names = {"select_36"};

    // initialize input data with values in [0.0, 1.0]
    // for (unsigned int i = 0; i < input_tensor_size; i++)
    //     input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    // create input tensor object from data values
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size,
        input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    auto output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                    &input_tensor, 1, output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    // assert(abs(floatarr[0] - 0.000045) < 1e-6);
    std::vector<float> results(1 * 518 * 518);
    for (unsigned i = 0; i < 1 * 518 * 518; i++)
    {
        results[i] = floatarr[i];
    }


    cv::Mat output_img(518,518,CV_32F,results.data());

    cv::resize(output_img, output_img, cv::Size(1216, 352), 0.0, 0.0,
               cv::INTER_CUBIC);
    double minVal, maxVal;
    cv::minMaxLoc(output_img, &minVal, &maxVal);
    output_img.convertTo(output_img, CV_32F);
    if (minVal != maxVal)
    {
        output_img = (output_img - minVal) / (maxVal - minVal);
    }
    output_img *= 255.0;
    output_img.convertTo(output_img, CV_8UC1);
    cv::applyColorMap(output_img, output_img, cv::COLORMAP_JET);

    cv::Mat concat_img;
    cv::vconcat(img, output_img, concat_img);

    while (true)
    {
        cv::imshow("pred image", concat_img);
        if (cv::waitKey(10) == 'q')
        {
            break;
        }
    }
    printf("Done!\n");
    return 0;
}