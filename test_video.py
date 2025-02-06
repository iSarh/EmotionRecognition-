import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('emotion_recognition_model.keras')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Path to the video file
video_path = r"C:\Users\Sarah\PycharmProjects\EmotionRecognition\test3.mp4"


# Open the video using OpenCV
cap = cv2.VideoCapture(video_path)


# Load the face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_skip = 2  # Speed up video by skipping every two frames
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frame to increase speed

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (48, 48))
        expanded_face = np.expand_dims(resized_face, axis=0)  # Add batch dimension
        normalized_face = expanded_face / 255.0 # Normalize pixel values

        # Predict emotions
        predictions = model.predict(normalized_face)
        predicted_class = np.argmax(predictions, axis=1)
        emotion = emotion_labels[predicted_class[0]]

        # Draw label and bounding box around the detected face
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Reduce wait time between frames to speed up video processing
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
