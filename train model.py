import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("--- [1] Starting Model Training ---")

# --- 1. Build a Lightweight Keras Model ---
def get_model():
    """Builds a lightweight CNN model, ideal for TFLite conversion."""
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)  # 10 output classes (for CIFAR-10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# --- 2. Train the Model ---
# Load CIFAR-10 dataset as a proxy for 'recyclable items'
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0 # Normalize

model = get_model()

print("Training... (This may take a minute)")
model.fit(train_images, train_labels, epochs=5, 
          validation_data=(test_images, test_labels),
          verbose=2)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"\nKeras Model Test Accuracy: {test_acc:.4f}")

# --- 3. Convert to TensorFlow Lite (TFLite) ---
print("\n--- [2] Converting to TFLite ---")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Optimizes for size
tflite_model = converter.convert()

# --- 4. Save the Final .tflite File ---
tflite_model_path = "model.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"\n--- [SUCCESS] ---")
print(f"Model trained and saved as '{tflite_model_path}'")
print("You can now run the Streamlit app.")
