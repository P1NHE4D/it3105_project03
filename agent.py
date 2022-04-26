import numpy as np
from keras.layers import Dense
from keras.models import Model, Sequential

from interface import Domain
import tensorflow as tf


class Agent:

    def __init__(
            self,
            domain: Domain,
            learning_rate=0.01,
            hidden_layers=None,
            weight_file=""
    ):
        self.domain = domain
        state, action = domain.get_init_state()
        input_shape = (len(state) + len(action),)
        if hidden_layers is None:
            hidden_layers = ["relu", 32]
        self.qnet = QNET(
            input_shape=input_shape,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            weight_file=weight_file
        )
        self.qnet_lite: LiteModel = LiteModel.from_keras_model(self.qnet)

    def train(self):
        # Net: maps given state and action to an evaluation
        # Policy: for all actions in the given state, pick the one with the highest evaluation

        pass

    def propose_action(self, state):
        pass


class QNET(Model):

    def __init__(self, input_shape, learning_rate, hidden_layers, weight_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_layers = [Dense(nodes, activation if activation != "linear" else None) for nodes, activation in hidden_layers]
        hidden_layers.append(Dense(1))
        self.model = Sequential(layers=hidden_layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.mse
        )

        self.model_trained = False
        try:
            self.load_weights(weight_file)
            self.model_trained = True
        except Exception as e:
            print("Unable to load weight file", e)

        # predict on a random sample to inform model of input size. Necessary to allow LiteModel to convert our model
        self.predict(np.random.random((1, *input_shape)))

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        return super().get_config()


class LiteModel:
    """
    Excluding this comment, this class was directly copied without modification from a IT3105 Blackboard thread titled
    "Found way to speed up Tensorflow by ~30x" authored by Mathias Pettersen. That thread references
    https://micwurm.medium.com/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98

    LiteModel provides a way to run small batches of predictions on a model more efficiently than by using the .predict
    or the .__call__ methods of the model directly. This comes at the cost of having to create a LiteModel version of
    the model any time it's weights are changed.
    """

    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]
