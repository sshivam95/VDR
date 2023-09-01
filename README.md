
# VDR: Virtual Dressing Room

This repository focuses on visual detection using different versions of the YOLO (You Only Look Once) object detection algorithm.

## Contents:
- **demo-yolo-v5.py**: A demonstration script showcasing object detection using YOLOv5.
- **demo.ipynb**: A Jupyter Notebook that provides a visual walkthrough of the detection process.
- **demo.py**: A generic demo script for object detection.
- **coco.names**: The list of object names that the YOLO model can detect, based on the COCO dataset.
- **yolov3.cfg**, **yolov4.cfg**: Configuration files for YOLOv3 and YOLOv4 respectively.
- **yolov5s-seg.pt**: Pre-trained weights for YOLOv5.

## Getting Started:

### Prerequisites:
Make sure you have Python installed on your system.

### Installation:
1. Clone this repository.
2. Navigate to the project directory.
3. Install the required Python packages using:
   ```
   pip install -r requirements.txt
   ```

### Usage:
1. To run the YOLOv5 demonstration, execute:
   ```
   python demo-yolo-v5.py
   ```
2. For a visual walkthrough, open and run the cells in `demo.ipynb`.
3. For a general demonstration, execute:
   ```
   python demo.py
   ```

### Notes:
- Ensure all the configuration and weights files are present in the directory when running the scripts.
- Performance might vary based on the specific YOLO model and weights used.
