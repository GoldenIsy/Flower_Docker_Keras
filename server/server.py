from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf

### Edit Manuel
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from clearml import Task

def main() -> None:
    
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    #task = Task.init(project_name="FedFischTest#4", task_name="AutoKeras_Server")
    
    
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(80, 80, 3), weights=None, classes=10
    )
    
    
   
    
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=3,
        min_eval_clients=2,
        min_available_clients=3,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server('localhost:8085', config={"num_rounds": 1}, strategy=strategy)


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    #(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    
    #Load Data from File
    #base_name = 'fish_pictures'
    #width = 80
    #data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    
    
    #Load Data from File
    #pfad = 'C:\Users\Manue\Desktop\docker\
    #base_name = 'fish_pictures'
    #width = 80
    #data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    #name = 'C:\Users\Manue\Desktop\docker\fish_pictures_80x80px.pkl'
    
    #Daten aus Docker Mount nehmen
    data = joblib.load(f'/app')
    
        #Prepare data
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    #Split Data to validate
    X_train, x_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=0.1, 
        shuffle=True,
        random_state=42,
    )
    
    
    
    # Use the last 5k training examples as a validation set
    #x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 80, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 80,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()