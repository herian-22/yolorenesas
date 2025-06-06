import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

dummy_input = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    "yolov5s_fp32.onnx",
    input_names=['images'],
    output_names=['output'],
    opset_version=12,
    dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}},
    do_constant_folding=True
)
print("Exported yolov5s_fp32.onnx with float32 input.")