import tensorflow as tf
import numpy as np
import struct


def save_weights_binary(model, output_file):
    with open(output_file, 'wb') as f:
        # Conv1 layers (6 5x5 filters)
        for w in model.layers[0].get_weights()[0]:
            for value in w.flatten():
                f.write(struct.pack('<f', float(value)))
        for b in model.layers[0].get_weights()[1]:
            f.write(struct.pack('<f', float(b)))

        # Conv2 layers (16 5x5 filters)
        for w in model.layers[2].get_weights()[0]:
            for value in w.flatten():
                f.write(struct.pack('<f', float(value)))
        for b in model.layers[2].get_weights()[1]:
            f.write(struct.pack('<f', float(b)))

        # FC layer (120x84)
        for w in model.layers[5].get_weights()[0].flatten():
            f.write(struct.pack('<f', float(w)))
        for b in model.layers[5].get_weights()[1]:
            f.write(struct.pack('<f', float(b)))

        # Output layer (84x10)
        for w in model.layers[7].get_weights()[0].flatten():
            f.write(struct.pack('<f', float(w)))
        for b in model.layers[7].get_weights()[1]:
            f.write(struct.pack('<f', float(b)))


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Create a simple CNN
model = tf.keras.Sequential([
    # First Convolutional Layer
    tf.keras.layers.Input(shape=(28, 28, 1)),  # Explicit input layer
    tf.keras.layers.Conv2D(6, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second Convolutional Layer
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten layer before dense layers
    tf.keras.layers.Flatten(),

    # Fully Connected Layers
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape for CNN
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Train
model.fit(x_train, y_train, epochs=5, validation_split=0.1, batch_size=128)

# Save weights in our custom binary format
save_weights_binary(model, 'mnist_weights.bin')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")