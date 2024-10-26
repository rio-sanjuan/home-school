This is an online meeting from 2024-09-15

## Notes

### Introduction

Based on book "Designing Machine Learning Systems" by Chip Huyen (available in local library)

### Serialization

"Converting ML models into a format that can be used by another application"
Jupyter Notebook/Colab are examples of this

##### Model Definition

Model architecture (number of hidden layers, output dim, etc.)

##### Model Parameters

The actual parameters of the model

```python
import torch

model = MyModel()
torch.save(model.state_dict(), 'model_weights.pth')
```

Tensorflow can save either or both

#### ML Deployment Myths

You can only deploy one or two ML models at a time…
If we don't do anything, model performance remains the same… (model drift)
You won't need to update your models…
Most MLE's don't need to worry about scale… (latency: time to serve prediction, can be a major barrier to scale)

### Batch Prediction (Asynchronous Prediction)

When predictions are generated periodically or whenever triggered. The predictions are stored somewhere, such as in SQL tables or an in-memory database, and retrieved as needed

Optimized for high throughput

### Online Prediction (Synchronous Prediction)

When predictions are generated and returned as soon as requests for these predictions arrive. (google translate)

Optimized for low latency

### Model Compression

The process of making a model smaller
"Inference Optimization" is increasing the speed of the inference step

#### Low-Rank Factorization

Specific to Convolutional Neural Networks

```python
# Import necessary libraries

import tensorflow as tf

from tensorflow.keras import layers, models

import numpy as np

  

# Function to count the number of parameters in a model

def count_params(model):

return np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])

  

# Define a model with standard convolutional layers

def create_standard_conv_model():

model = models.Sequential([

layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),

layers.MaxPooling2D((2, 2)),

layers.Conv2D(64, (3, 3), activation='relu'),

layers.MaxPooling2D((2, 2)),

layers.Flatten(),

layers.Dense(64, activation='relu'),

layers.Dense(10, activation='softmax')

])

return model

  

# Define a model with depthwise separable convolutional layers

def create_depthwise_separable_conv_model():

model = models.Sequential([

layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),

layers.MaxPooling2D((2, 2)),

layers.SeparableConv2D(64, (3, 3), activation='relu'),

layers.MaxPooling2D((2, 2)),

layers.Flatten(),

layers.Dense(64, activation='relu'),

layers.Dense(10, activation='softmax')

])

return model

  

# Create models

standard_model = create_standard_conv_model()

depthwise_model = create_depthwise_separable_conv_model()

  

# Compile models to ensure they are fully built

standard_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

depthwise_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  

# Count the parameters in each model

standard_model_params = count_params(standard_model)

depthwise_model_params = count_params(depthwise_model)

  

print(f"Number of parameters in the standard convolution model: {standard_model_params / 1e6:.2f} million")

print(f"Number of parameters in the depthwise separable convolution model: {depthwise_model_params / 1e6:.2f} million")

  

# Optional: Print model summaries for detailed parameter breakdown

print("\nStandard Convolution Model Summary:")

standard_model.summary()

  

print("\nDepthwise Separable Convolution Model Summary:")

depthwise_model.summary()
```

#### Knowledge Distillation

A method in which a small model (student) is trained to mimic a larger model or ensemble of model (teacher). You deploy the smaller model

#### Quantization

Using smaller memory-intensive variables for storing model parameters

```python
# Import necessary libraries

import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.datasets import mnist

import numpy as np

import os

  

# Load and preprocess the MNIST dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalize to [0,1]

  

# Define a simple neural network model

def create_model():

model = models.Sequential([

layers.Flatten(input_shape=(28, 28)), # Flatten the 28x28 images to a vector

layers.Dense(128, activation='relu'), # Dense layer with ReLU activation

layers.Dropout(0.2), # Dropout layer to reduce overfitting

layers.Dense(10, activation='softmax') # Output layer with 10 classes (digits 0-9)

])

model.compile(optimizer='adam',

loss='sparse_categorical_crossentropy',

metrics=['accuracy'])

return model

  

# Instantiate and train the model

model = create_model()

model.fit(x_train, y_train, epochs=5, validation_split=0.1, batch_size=32)

  

# Evaluate the model

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test accuracy before quantization: {test_acc:.4f}")

  

# Save the original model to a file

original_model_path = 'model_original.h5'

model.save(original_model_path)

  

# Get size of the original model

original_model_size = os.path.getsize(original_model_path) / (1024 * 1024) # Convert bytes to MB

print(f"Size of the original model: {original_model_size:.2f} MB")

  

# Convert the model to TensorFlow Lite format

converter = tf.lite.TFLiteConverter.from_keras_model(model)

  

# Set the optimization strategy to optimize for size

converter.optimizations = [tf.lite.Optimize.DEFAULT]

  

# Convert the model

tflite_model = converter.convert()

  

# Save the quantized model to a file

quantized_model_path = 'model_quantized.tflite'

with open(quantized_model_path, 'wb') as f:

f.write(tflite_model)

  

# Get size of the quantized model

quantized_model_size = os.path.getsize(quantized_model_path) / (1024 * 1024) # Convert bytes to MB

print(f"Size of the quantized model: {quantized_model_size:.2f} MB")

  

# Calculate and print space saved

space_saved = original_model_size - quantized_model_size

print(f"Space saved by quantization: {space_saved:.2f} MB")

  

# Optional: Load and run inference with the quantized model

# Create an interpreter for the quantized model

interpreter = tf.lite.Interpreter(model_content=tflite_model)

interpreter.allocate_tensors()

  

# Get input and output tensor details

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

  

# Prepare the test input (just the first image from the test set for demonstration)

test_input = np.expand_dims(x_test[0], axis=0).astype(np.float32)

  

# Set the tensor to the input

interpreter.set_tensor(input_details[0]['index'], test_input)

  

# Run inference

interpreter.invoke()

  

# Get the result

output = interpreter.get_tensor(output_details[0]['index'])

print(f"Prediction for the first test image: {np.argmax(output)}")

  

# Optional: Compare with the actual label

print(f"Actual label: {y_test[0]}")
```

## Future Topics to Explore

1. Difference between "training" and "inference"
2. CentML (Model Compression)
3. Low-Rank Factorization

## Attendees

* #HetavPandya (Presenter)
* Anton
* Sergey Semernev
* Kacy Chou
