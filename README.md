Vehicle Detection and Color Classification using YOLOv8 and ResNet50
Overview

This project focuses on vehicle detection and color classification using deep learning techniques. We utilize a top-view vehicle detection image dataset and a pretrained YOLOv8 model for vehicle type detection. Additionally, for accurate color classification, we integrate our original dataset with the V-COR dataset. A pretrained ResNet50 model is used for color classification. Finally, both models are combined to provide a comprehensive output displaying vehicle type and color.

Dataset

1.Vehicle Type Detection: A top-view vehicle detection image dataset is used.

2.Color Classification: The V-COR dataset is merged with our original dataset to enhance color classification accuracy.

Models Used

1.Vehicle Type Detection: YOLOv8 (pretrained model)

2.Color Classification: ResNet50 (pretrained model)

3.Final Output: Integration of both models to detect vehicle type and color together.

Vehicle Detection using YOLOv8

1.Loading the pretrained YOLOv8 model.

2.Training/fine-tuning on the dataset.

3.Performing inference for vehicle type detection.

4.Color Classification using ResNet50

5.Loading the pretrained ResNet50 model.

6.Training/fine-tuning for color classification.

7.Classifying vehicle colors accurately.

8.Integration of Models

9.Combining YOLOv8 and ResNet50 models.

10.Displaying final output with detected vehicle type and color.

Dependencies

Python 3.x

PyTorch

Ultralytics YOLOv8

TensorFlow/Keras (for ResNet50)

OpenCV

NumPy

Pandas

How to Run

Clone the repository:

git clone https://github.com/your-username/vehicle-detection-yolov8.git
cd vehicle-detection-yolov8

Install dependencies:

pip install -r requirements.txt

Results

The model successfully detects vehicles and classifies them into different types.

The color classification model enhances accuracy by leveraging the V-COR dataset.

The integrated model displays both vehicle type and color in the final output.

Future Improvements

Enhance dataset quality by including more diverse vehicle images.

Improve model accuracy using hyperparameter tuning.

Implement real-time inference with optimized performance.
