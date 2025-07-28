# RetinaFace Face Recognition System

This repository provides a PyTorch implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641) and a pipeline for face detection and recognition using ONNX and PyTorch.

## Features

- Face detection using RetinaFace (ResNet50 or MobileNet0.25 backbone)
- Face recognition using FaceNet (ONNX)
- Video processing to find and timestamp appearances of a reference face

## Folder Structure

```
Pytorch_Retinaface/
│
├── face.py                
├── detect.py              
├── facenet512.onnx        
├── weights/
│   └── Resnet50_Final.pth 
├── data/
│   └── config.py          
├── models/
│   └── retinaface.py
|   └── net.py      
├── layers/
│   └── functions/
│       └── prior_box.py
├── utils/
│   └── box_utils.py
├── examples/                # Place your images and videos here
│   ├── 4.jpeg
│   ├── 4.mp4
│   └── ...
└── README.md
```

## Installation

1. **Clone the repository**
    ```sh
    git clone https://github.com/biubug6/Pytorch_Retinaface.git
    cd Pytorch_Retinaface
    ```

2. **Install dependencies**
    ```sh
    pip install torch torchvision onnxruntime opencv-python tqdm numpy
    ```

3. **Download Pretrained Weights**
    - Ensure `weights/Resnet50_Final.pth` is present. If not, download from the official [RetinaFace model zoo](https://github.com/biubug6/Pytorch_Retinaface#training) or [Google Drive link](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1).
    - Ensure `facenet512.onnx` is present in the root directory.

## Usage

### 1. Prepare Your Data

- Place your reference image (e.g., `examples/1.jpg`) and video file (e.g., `examples/1.mp4`) in the `examples/` folder.

### 2. Run Face Recognition

#### **To use GPU (default, if available):**
```sh
python face.py --image images/1.jpg --video images/1.mp4
```
- The script will automatically use GPU if a CUDA-capable device and CUDA-enabled PyTorch are installed.

#### **To force CPU usage:**
On **Windows**:
```sh
set CUDA_VISIBLE_DEVICES=
python face.py --image images/1.jpg --video images/1.mp4
```
On **Linux/macOS**:
```sh
CUDA_VISIBLE_DEVICES= python face.py --image images/1.jpg --video images/1.mp4
```
Or, you can edit [`face.py`](face.py) and set:
```python
DEVICE = "cpu"
```

- `--image`: Path to the reference face image.
- `--video`: Path to the video file to process.

### 3. Output

- The script will print:
    - Number of frames where the reference face was detected
    - Average detection probability
    - First and last appearance timestamps
    - All appearance intervals in the video

## Notes

- By default, RetinaFace uses the ResNet50 backbone. You can change this in [`face.py`](face.py) by modifying `RETINAFACE_NETWORK`.
- Detection and recognition thresholds can be adjusted in [`face.py`](face.py) for your use case.
- For best results, use clear, frontal reference images.