import argparse
import os

import numpy as np
import tensorflow as tf

import flwr as fl


### Edit
import joblib
import math
from sklearn.model_selection import train_test_split
from os import listdir
import time

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--grpc_server_address",
        type=str,
        default="[::]",
        help="gRPC server address (default: [::])",
    )
    parser.add_argument(
        "--grpc_server_port",
        type=int,
        default=8080,
        help="gRPC server port (default: 8080)",
    )
    parser.add_argument(
        "--partition", type=int, required=False, default=1, help="Partition index (no default)"
    )
    parser.add_argument(
        "--clients", type=int, required=False, default=10, help="Number of clients (no default)",
    )
    args = parser.parse_args()

    # Load and compile Keras model
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(80, 80, 3), weights=None, classes=10
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


    # Load a subset of Data to simulate the local data partition
    print("Partition:" + str(args.partition) + "Clients: " + str(args.clients))
     
    
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition, args.clients)
    print("Partition:" + str(args.partition) + " geladen.")

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)
   
   
    #Server Connection
    #fl.app.start_client(args.grpc_server_address, args.grpc_server_port, client)
    fl.client.start_numpy_client('localhost:8085', client=client)

def load_partition(idx: int, numClients: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    
    #Load Data from File
    #base_name = 'fish_pictures'
    #width = 80
    #data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    
    #Daten aus Docker Mount nehmen
    data = joblib.load(f'/app')
    
    #Prepare data
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    #Split Data in Test&Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        shuffle=True,
        random_state=42,
    )

    assert idx in range(numClients)
    
    
    #Ganzes Array durch Anzahl der Clients teilen
    partition_size = math.floor(len(X_train)/numClients)
    #Test ist 20% der ganzen Partition
    test_size = math.floor(partition_size/5)
    
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        X_train[idx * partition_size : (idx + 1) * partition_size],
        y_train[idx * partition_size : (idx + 1) * partition_size],
    ), (
        X_test[idx * test_size : (idx + 1) * test_size],
        y_test[idx * test_size : (idx + 1) * test_size],
    )


if __name__ == "__main__":
    main()
