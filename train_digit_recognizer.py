import numpy as np
import cv2
import os
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Paths
mnist_path = 'D:/HWDR'
devanagari_path = 'D:/HWDR/devanagari_digits'
custom_images_path = 'D:/HWDR/custom_images'  # Custom images folder

# Load MNIST dataset
(train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = mnist.load_data()

# Preprocess MNIST dataset
train_images_mnist = train_images_mnist.reshape((train_images_mnist.shape[0], 28, 28, 1)).astype('float32') / 255
test_images_mnist = test_images_mnist.reshape((test_images_mnist.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels_mnist = to_categorical(train_labels_mnist, 10)
test_labels_mnist = to_categorical(test_labels_mnist, 10)

# Function to load Devanagari dataset
def load_devanagari_data(path):
    images, labels = [], []
    for digit in range(10):
        digit_path = os.path.join(path, str(digit))
        for img_name in os.listdir(digit_path):
            img_path = os.path.join(digit_path, img_name)
            img = Image.open(img_path).convert('L').resize((28, 28))
            img = np.array(img)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            images.append(img / 255.0)
            labels.append(digit)
    return np.array(images).reshape((-1, 28, 28, 1)), to_categorical(np.array(labels), 10)

# Load Devanagari dataset
images_devanagari, labels_devanagari = load_devanagari_data(devanagari_path)

# Function to load custom images
def load_custom_images(path):
    images, labels = [], []
    for digit in range(10):
        digit_path = os.path.join(path, str(digit))
        if not os.path.exists(digit_path):
            continue
        for img_name in os.listdir(digit_path):
            img_path = os.path.join(digit_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = img / 255.0
            images.append(img)
            labels.append(digit)
    return np.array(images).reshape((-1, 28, 28, 1)), to_categorical(np.array(labels), 10)

# Load custom images
custom_images, custom_labels = load_custom_images(custom_images_path)

# Combine all datasets
train_images_mnist = np.concatenate((train_images_mnist, custom_images), axis=0)
train_labels_mnist = np.concatenate((train_labels_mnist, custom_labels), axis=0)

# Data augmentation for Devanagari images
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(images_devanagari)

# Define Optimized CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

# Train MNIST model with custom images
mnist_model = build_model()
mnist_model.fit(train_images_mnist, train_labels_mnist, epochs=10, batch_size=32, callbacks=[early_stopping])
mnist_model.save(os.path.join(mnist_path, 'mnist_model_updated.h5'))
print("Updated MNIST model trained and saved as 'mnist_model_updated.h5'")

# Evaluate MNIST model
mnist_loss, mnist_accuracy = mnist_model.evaluate(test_images_mnist, test_labels_mnist)
print(f'MNIST Accuracy: {mnist_accuracy * 100:.2f}%')

# Train Devanagari model
devanagari_model = build_model()
devanagari_model.fit(datagen.flow(images_devanagari, labels_devanagari, batch_size=32), epochs=10, callbacks=[early_stopping])
devanagari_model.save(os.path.join(mnist_path, 'devanagari_model.h5'))
print("Devanagari model trained and saved as 'devanagari_model.h5'")

# Evaluate Devanagari model
deva_loss, deva_accuracy = devanagari_model.evaluate(images_devanagari, labels_devanagari)
print(f'Devanagari Accuracy: {deva_accuracy * 100:.2f}%')

# See Size of the Dataset
# Print dataset sizes
print(f"MNIST Training dataset size: {train_images_mnist.shape[0]} images")
print(f"Devanagari dataset size: {images_devanagari.shape[0]} images")
print(f"Custom dataset size: {custom_images.shape[0]} images")

# Total dataset size
total_size = train_images_mnist.shape[0] + images_devanagari.shape[0]
print(f"Total dataset size (MNIST + Devanagari + Custom): {total_size} images")

