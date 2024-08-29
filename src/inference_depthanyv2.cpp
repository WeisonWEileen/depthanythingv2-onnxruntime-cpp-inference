#include <fstream>
#include <string>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

int main()
{

    std::string config_file = "config/config.yaml";
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);


    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Env env;
    Ort::Session session{env, ORT_TSTR("data/models/vitb.onnx"),
                         Ort::SessionOptions{nullptr}};

    // Allocate model inputs: fill in shape and size
    std::array<float, ...> input{};
    std::array<int64_t, ...> input_shape{...};
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(),
                                        input_shape.data(), input_shape.size());
    const char* input_names[] = {...};

    // Allocate model outputs: fill in shape and size
    std::array<float, ...> output{};
    std::array<int64_t, ...> output_shape{...};
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info, output.data(), output.size(), output_shape.data(),
        output_shape.size());
    const char* output_names[] = {...};

    // Run the model
    session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                 output_names, &output_tensor, 1);
}