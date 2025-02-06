import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input


# Path to the directory containing images categorized into different emotion folders
train_dir = "C:/Users/Sarah/PycharmProjects/EmotionRecognition/train"

# Create an instance of ImageDataGenerator to automatically load and categorize images based on folders
train_datagen = ImageDataGenerator(rescale=1./255)   # Normalize pixel values between 0 and 1

# Load data using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # Resize images to fit the neural network input
    batch_size=32,         # Batch size
    class_mode='categorical'  # Multi-class classification
)

# Convert train_generator to tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,  # Uses the image generator data
    output_signature=(
        tf.TensorSpec(shape=(32, 48, 48, 3), dtype=tf.float32),  # Data shape (32 images of 48x48 with 3 channels)
        tf.TensorSpec(shape=(32, 7), dtype=tf.float32)   # Labels shape (32 labels with 7 emotion categories)
    )
)

# Repeat dataset indefinitely
train_dataset = train_dataset.repeat()  # تكرار البيانات


# Build the neural network model
model = models.Sequential([
    Input(shape=(48, 48, 3)),   # Input layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 emotion categories
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using the dataset
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # عدد الخطوات (الباتشات) في كل دورة تدريبية
    epochs=10  # Number of training epochs
)

# Save the trained model
model.save("emotion_recognition_model.keras")
