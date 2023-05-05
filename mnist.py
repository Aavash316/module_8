import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow import keras

# Start an MLflow run to track the experiment
mlflow.start_run()

# Load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Set the model parameters
learning_rate = 0.01
batch_size = 128
epochs = 2

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Log all the parameters to MLflow
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", epochs)

# Train the model and log the metrics to MLflow using the KerasCallback

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_test, y_test))

# Log the final metrics to MLflow
mlflow.log_metric("train_loss", history.history["loss"][-1])
mlflow.log_metric("train_acc", history.history["accuracy"][-1])
mlflow.log_metric("val_loss", history.history["val_loss"][-1])
mlflow.log_metric("val_acc", history.history["val_accuracy"][-1])

# Save the model to a TensorFlow SavedModel format and log it to MLflow
model_save_path = "model"
tf.saved_model.save(model, model_save_path)


# End the MLflow run
mlflow.end_run()