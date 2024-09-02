# Depth-Anything v2 OnnxRuntime c++ 部署教程

### Intro
[Depth-Anythin v2](https://github.com/DepthAnything/Depth-Anything-V2) 的源码开源之后，3 months 已经收货了 3k 的star数，可谓是单目深度估计的最火热的模型之一，此深度估计方法可作为一些slam 任务和 embodied AI task 的 depth input。
因此使用 OnnxRuntime, TensorRT 等框架对模型进行加速提高帧率是有意义的。本文将介绍使用 OnnxRuntime 对作者提供的 Vitb 模型进行部署。笔者在部署过程中踩了一些坑，在这里记录下来，希望可以帮助其他人。

### 导出 Onnx 模型
github已经有了作者写好了官方的pth模型转onnx模型，[传送门](https://github.com/fabio-sim/Depth-Anything-ONNX/tree/main)。可以直接运行 ```export.py```，选择想要的模型文件。我选择导出的是 vitb 模型（参数量为371m，不至于太大,精度不至于太低）
```
python3 export.py --model b --opset 18 --precision float32
```
### 数据预处理 workflow
```mermaid {align="center"}
flowchart TD
    opencv读取 --> 2["Resize 图像到 (518,518)"]--> BRG转RGB --> 1[3通道像素值映射到0~1的float32] --> 6[数据搬运至 std::vector<float>] --> 5[使用cv::dnn::blobFromImage 使得维度变成 1×3×518×518 ]--> 7[创建 Ort::Value 输入模型推理]
```

### 模型推理 workflow


```mermaid {align="center"}
flowchart TD
subgraph 模型参数设置与初始化
    A[初始化] --> B[设置env]
    B --> C[设置session选项]
    C --> D[设置session]
    D --> E[设置input]
    E --> F[设置output]
    end
subgraph 模型前向推理
    A --> G[运行模型]
    G --> H[设置输入Tensor]
    H --> I[执行session]
    I --> J[获取输出Tensor]
end
subgraph free
    G --> K[释放模型]
    K --> L[释放资源]
end

```


### 使用 CUDA 加速
OnnxRuntime 加速提供的CUDA加速接口非常方便，只需要在 ```sessionOptions``` 中加上CUDA对应选项即可。
## Appendix
