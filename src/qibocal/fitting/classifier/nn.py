from pathlib import Path

import keras_tuner as kt  # pylint: disable=import-error
import numpy as np
from tensorflow.keras import backend as K  # pylint: disable=import-error
from tensorflow.keras import callbacks, optimizers  # pylint: disable=import-error
from tensorflow.keras.layers import (  # pylint: disable=import-error
    BatchNormalization,
    Dense,
    Layer,
)
from tensorflow.keras.models import (  # pylint: disable=import-error
    Sequential,
    load_model,
)

from .utils import identity


def constructor(hyperpars):
    r"""Return the model class.

    Args:
        _hyperparams: Model hyperparameters.
    """
    best_hps = kt.HyperParameters.from_config(hyperpars)
    tuner = kt.Hyperband(_hypermodel)

    return tuner.hypermodel.build(best_hps)


def hyperopt(x_train, y_train, path):
    r"""Perform an hyperparameter optimization and return the hyperparameters.

    Args:
        x_train: Training inputs.
        y_train: Training outputs.
        path (path): Model save path.

    Returns:
        Dictionary with model's hyperparameters.
    """
    tuner = kt.Hyperband(
        _hypermodel,
        objective="val_accuracy",
        max_epochs=150,
        directory=path,
        project_name="NNmodel",
    )
    tuner.search_space_summary()

    stop_early = callbacks.EarlyStopping(monitor="val_accuracy", patience=10)
    tuner.search(
        x_train, y_train, epochs=120, validation_split=0.2, callbacks=[stop_early]
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps.get_config()


normalize = identity


def dump(model, save_path: Path):
    r"""Dumps the `model` in `save_path`"""
    model.save(save_path)


def predict_from_file(loading_path: Path, input: np.typing.NDArray):
    r"""This function loads the model saved in `loading_path`
    and returns the predictions of `input`.
    """
    model = load_model(loading_path)
    return model.predict(input)


class RBFLayer(Layer):
    r"""Deploy a neural network layer with RBF activation function.

    Args:
        units: Number of nodes in the layer.
        gamma: Activation function's parameter.
    """

    def __init__(self, units, gamma, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(
            name="mu",
            shape=(int(input_shape[1]), self.units),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


def _hypermodel(hp):
    r"""Hyper model for the Keras tuner"""
    hp_units_1 = hp.Int("units_1", min_value=16, max_value=1056, step=16)
    hp_units_2 = hp.Int("units_2", min_value=16, max_value=1056, step=16)
    activation = hp.Choice("activation", ["relu", "sigmoid", "tanh", "RBF"])
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2)
    optimizer_choice = hp.Choice("optimizer", ["Adam", "Adagrad", "SGD", "RMSprop"])
    losses = hp.Choice("losses", ["binary_crossentropy", "categorical_crossentropy"])

    model = Sequential()
    model.add(BatchNormalization())
    if activation == "RBF":
        model.add(
            RBFLayer(
                hp_units_1,
                hp.Float("mu", min_value=1e-4, max_value=1),
                input_shape=(2,),
            )
        )
    else:
        model.add(Dense(hp_units_1, input_shape=(2,), activation=activation))

    if activation == "RBF":
        model.add(
            RBFLayer(
                hp_units_2,
                hp.Float("mu", min_value=1e-4, max_value=1),
                input_shape=(2,),
            )
        )
    else:
        model.add(Dense(hp_units_2, input_shape=(2,), activation=activation))

    model.add(Dense(1, activation="sigmoid"))

    if optimizer_choice == "Adam":
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == "Adagrad":
        optimizer = optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_choice == "SGD":
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_choice == "RMSprop":
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError

    model.compile(optimizer=optimizer, loss=losses, metrics=["accuracy"])
    return model
