# Real-time Object Detection with YOLOv5 and GTK

This project implements real-time object detection using YOLOv5 and displays the results in a GTK window. It uses ONNX Runtime for inference and OpenCV for image processing.

## Features

- Real-time object detection using YOLOv5
- GTK-based GUI for displaying camera feed
- Performance statistics (FPS, inference time, CPU cores)
- Bounding box visualization with confidence scores

## Dependencies

- OpenCV 4.x
- GTK+ 3.0
- ONNX Runtime
- C++17 or later

## Building

1. Install dependencies:
```bash
sudo apt-get install libopencv-dev libgtk-3-dev
```

2. Install ONNX Runtime:
```bash
# Download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases
# Extract to /usr/local/onnxruntime
```

3. Compile:
```bash
cd src
make
```

## Running

```bash
./camera_gtk
```

## Model Export

To export YOLOv5 model to ONNX format with float32 input:

```bash
python3 export_yolov5.py
```

## License

MIT License 