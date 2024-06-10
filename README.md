# Roadside-Object-Detection-YOLOv8
This is a project for my deep learning class. Here YOLO v8 is used to detect and recognize common roadside objects. 

 Roadside Object Detection with YOLOv8

Overview

This project aims to detect and recognize common objects seen on roadsides using the YOLOv8 (You Only Look Once version 8) object detection model. The project involves training various YOLOv8 model variants on a dataset of annotated roadside images, evaluating their performance, and selecting the best model based on key metrics.

Table of Contents

-Overview
- Features
- Installation
- Usage
- Dataset
- Model Training
- Evaluation
- Results
- Contributing

Features

- Detection of multiple object classes such as cars, heavy vehicles, workers, and pedestrians.
- Utilizes YOLOv8 models for real-time object detection.
- Comprehensive performance evaluation using precision, recall, mAP, and IoU metrics.

Installation

To set up this project locally, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/roadside-object-detection-yolov8.git
   cd roadside-object-detection-yolov8
   ```

2. Install required packages:
   ```sh
   pip install -r requirements.txt
   ```

Usage

Accessing the Dataset

The dataset is sourced from RoboFlow. To access it, follow these steps:

1. **Sign up for RoboFlow**:
   [RoboFlow](https://roboflow.com/)
   
2. **Download the dataset**:
   Use the RoboFlow API to download and preprocess the dataset.
   
   ```python
   from roboflow import Roboflow
   
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace().project("your-project-name")
   dataset = project.version("1").download("yolov8")
   ```

Training the Model

To train the model, run the following script:

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model.train(data='path/to/your/data.yaml', epochs=30, imgsz=640, batch=16, ...)
```

Evaluating the Model

Evaluate the trained model using:

```python
results = model.val()
```

Dataset

The dataset consists of 2,454 images with approximately 7,500 annotations of common roadside objects. It is split into:
- Training set: 80% (1,963 images)
- Validation set: 10% (246 images)
- Test set: 10% (245 images)

Annotations include bounding boxes for each object, with classes such as cars, heavy vehicles, workers, and pedestrians.

Model Training

Several YOLOv8 model variants were trained using different configurations and hyperparameters. The training involved:
- Initializing the YOLOv8 model with pre-trained weights.
- Applying data augmentation techniques.
- Using early stopping based on validation loss.

Evaluation

The performance of the models was evaluated using:
- Precision
- Recall
- mean Average Precision (mAP)
- Intersection over Union (IoU)

Results

The results of the model training and evaluation are as follows:

- Model 1: Trained for 10 epochs with an overall mAP50 of 74.5%.
- Model 2: Trained for 20 epochs with an overall mAP50 of 76.9%.
- Model 3: Trained for 30 epochs with an overall mAP50 of 77.7%.
- Model 4: Trained for 30 epochs with an overall mAP50 of 80.4%.

Contributing

Contributions are welcome! Please fork this repository and submit pull requests for any enhancements or bug fixes.
