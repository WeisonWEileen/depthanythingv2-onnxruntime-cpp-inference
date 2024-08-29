import torch
import torchvision

dummy_input = torch.randn(1, 1, 352, 1216, device="cuda")  # 修改这里

model = torch.load(
    "/home/weison/Desktop/ONNX-Runtime-Inference/data/models/depth_anything_v2_vitb.pth"
)

input_names = ["inputgdb"]
output_names = ["pred"]

torch.onnx.export(
    model,
    dummy_input,
    "depth_anything_v2_vitb.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
)
