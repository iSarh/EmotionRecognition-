# Emotion Recognition from Video

## Project Overview

This project focuses on real-time emotion recognition from video using deep learning techniques. The system utilizes a Convolutional Neural Network (CNN) trained on facial expression data to classify emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The model is implemented using TensorFlow and Keras, and add is used for video processing.

## Features

- Real-time emotion detection from video.
- Pre-trained deep learning model for accurate predictions.
- Utilizes OpenCV for video frame extraction and preprocessing.
- Supports grayscale image conversion to improve model performance.
- Displays the detected emotion label on the video frames.

## Installation

### Prerequisites

Ensure you have Python 3.11 installed along with the necessary dependencies.

### Install Dependencies

Run the following command to install the required packages:

```bash
pip install tensorflow keras opencv-python numpy
```

## Dataset

The model is trained on a dataset of 28,709 facial images belonging to seven emotion classes. The images are preprocessed to a size of 48x48 pixels.

## Training the Model

To train the model, run:

```bash
python train_model.py
```

The trained model will be saved in the project directory as `emotion_recognition_model.keras`.

## Testing with Video

To test the model on a video file, run:

```bash
python test_video.py
```

Make sure to update the video path inside `test_video.py` accordingly.

## Model Performance

The model achieves an accuracy of approximately 76% after 10 epochs of training. Performance may vary based on dataset quality and preprocessing techniques.

## The Result

The model provides real-time emotion predictions on video frames. However, accuracy may depend on lighting conditions, facial expressions, and video quality. Further optimization can improve prediction stability and precision.

https://github.com/user-attachments/assets/69342333-2d4f-4504-a5c1-818002c2c4c5


## Future Improvements

- Enhance accuracy by using a more diverse and larger dataset.
- Optimize model architecture for better real-time performance.
- Implement real-time webcam-based emotion recognition.
- Improve frame processing speed for smoother video playback.

## License

This project is open-source and available for educational and research purposes.

