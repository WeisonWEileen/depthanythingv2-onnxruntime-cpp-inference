#include "include/dep_ang_v2_onnx.hpp"
#include <filesystem>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

int main()
{

    std::string configfile_path = "config/config.yaml";
    cv::FileStorage configs(configfile_path.c_str(), cv::FileStorage::READ);

    std::string model_path = configs["onnx_model_path"];
    std::string picture_path = configs["picture_path"];

    if (!std::filesystem::exists(model_path))
    {
        std::cout << "Model file does not exist: " << model_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "Load model: " << model_path << std::endl;

    if (!std::filesystem::exists(picture_path))
    {
        std::cout << "Picture file does not exist: " << picture_path
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "Load pict: " << picture_path << std::endl;

    DepthAnyV2Onnx model(model_path.c_str());
    cv::Mat img = cv::imread(picture_path);
    std::cout << "img dims: {" << img.rows << " " << img.cols << "} " << std::endl;
    cv::Mat pred = model.predict(img);

    while (true)
    {
        cv::imshow("pred image", pred);
        if (cv::waitKey(10) == 'q')
        {
            break;
        }
    }

    //     auto memory_info =
    //         Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // Ort::Env env;
    // Ort::Session session{env, ORT_TSTR(mode_path.c_str()),
    //                      Ort::SessionOptions{nullptr}};

    // // Allocate model inputs: fill in shape and size
    // std::array<float, ...> input{};
    // std::array<int64_t, ...> input_shape{...};
    // Ort::Value input_tensor =
    //     Ort::Value::CreateTensor<float>(memory_info, input.data(),
    //     input.size(),
    //                                     input_shape.data(),
    //                                     input_shape.size());
    // const char* input_names[] = {...};

    // // Allocate model outputs: fill in shape and size
    // std::array<float, ...> output{};
    // std::array<int64_t, ...> output_shape{...};
    // Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
    //     memory_info, output.data(), output.size(), output_shape.data(),
    //     output_shape.size());
    // const char* output_names[] = {...};

    // // Run the model
    // session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
    //              output_names, &output_tensor, 1);
}