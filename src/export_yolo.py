from ultralytics import YOLO
import torch

def export_yolo_to_onnx():
    # Load model YOLOv8
    model = YOLO('yolov8n.pt')  # atau gunakan model yang sudah dilatih 'best.pt'
    
    # Export ke ONNX dengan konfigurasi khusus untuk OpenCV
    success = model.export(format='onnx',  # format output
                         imgsz=640,        # ukuran input
                         half=False,       # gunakan FP32
                         simplify=True,    # optimasi model
                         opset=12,         # versi ONNX opset
                         dynamic=False)    # non-dynamic batch size
    
    if success:
        print("Model berhasil diekspor ke ONNX!")
    else:
        print("Gagal mengekspor model!")

if __name__ == "__main__":
    export_yolo_to_onnx()

import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
model.export(format='onnx', imgsz=640, simplify=True, dynamic=False, opset=12)
print("Model berhasil diekspor ke yolov5s.onnx") 