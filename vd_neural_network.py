import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random, nn

from jax.tree_util import register_pytree_node_class

import numpy as np
import time
import math
import copy
from functools import partial

# Constants
# [x y yaw vx vy yaw_rate steering Fx]
l_bounds = jnp.array((-0.5, -0.5, -0.5, 0.0, -5.0, -0.35, -0.26, -1000))
u_bounds = jnp.array((0.5, 0.5, 0.5, 20.0, 5.0, 0.35, 0.26, 1000))

# [x y yaw vx vy yaw_rate]
l_out_bounds = jnp.array((-0.5, -0.5, -0.5, 0.0, -5.0, -0.35))
u_out_bounds = jnp.array((0.5, 0.5, 0.5, 20.0, 5.0, 0.35))

## Helper classes

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def input_scaler(x_input):
    # Scales 0 to 1
    x_scaled = jnp.divide((x_input - l_bounds), (u_bounds - l_bounds))
    return x_scaled

def output_scaler(y_input):
    # Scales 0 to 1
    y_scaled = jnp.divide((y_input - l_out_bounds), (u_out_bounds - l_out_bounds))
    return y_scaled

def output_descalar(y_scaled):
    y_output = jnp.multiply(y_scaled, (u_out_bounds-l_out_bounds)) + l_out_bounds
    return y_output

def get_datasets(data):
    n_rows = data.shape[0]
    np.random.shuffle(data)
    train_size = int(n_rows * 0.8)
    training, test = data[:train_size, :], data[train_size:, :]
    return training[:, :8], training[:, 8:], test[:, :8], test[:, 8:]

# Model class
# Receives inputs
# Container for layers and params
# Note that variable members are stored as children
# and they will be passed to the class when the pytree is unflattened
@register_pytree_node_class
class VehicleDynamicsNN:
    def __init__(self, layer_sizes, parameters = None):
        self.layer_sizes = layer_sizes
        self.nn_parameters = parameters
        if (not parameters) and self.layer_sizes:
            self.initialize_params()

    def initialize_params(self):
        initializer = nn.initializers.normal(0.01)
        self.nn_parameters = [
            [initializer(random.PRNGKey(42), (n, m), dtype=jnp.float32),
             initializer(random.PRNGKey(42), (n,), dtype=jnp.float32)]
            for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

    def init_network_params(self):
        key = random.PRNGKey(0)
        keys = random.split(key, len(self.layer_sizes))
        self.nn_parameters = [random_layer_params(m, n, k) for m, n, k in zip(self.layer_sizes[:-1], self.layer_sizes[1:], keys)]

    def tree_flatten(self):
        children = ( self.layer_sizes, self.nn_parameters)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # Vectorized forward pass
    @partial(vmap, in_axes=(None, None, 0))
    def forward(self, nn_params, x):
        x_scaled = input_scaler(x)
        x_scaled_ori = copy.deepcopy(x_scaled)
        for w, b in nn_params[:-1]:
            outputs = jnp.dot(w, x_scaled) + b
            x_scaled = nn.relu(outputs)
        final_w, final_b = nn_params[-1]
        # Residuals
        x_scaled = x_scaled_ori + x_scaled
        final_output = jnp.dot(final_w, x_scaled) + final_b
        return final_output

    # Alternative way of vectorising the forward pass. Arguably
    # better so there is no need for two methods for forward pass
    #@partial(vmap, in_axes=(0))
    # def batched_forward(self, params, x):
    #     return vmap(self.forward, in_axes=(None, 0))(params, x)

    # For some reason a non-vectorized version forward is needed.
    def predict(self, x):
        # Per-example predictions
        x_scaled = input_scaler(x)
        x_scaled_ori = copy.deepcopy(x_scaled)
        for w, b in self.nn_parameters[:-1]:
            outputs = jnp.dot(w, x_scaled) + b
            x_scaled = nn.relu(outputs)
        final_w, final_b = self.nn_parameters[-1]
        # Residuals
        x_scaled = x_scaled_ori + x_scaled
        final_output = jnp.dot(final_w, x_scaled) + final_b
        return final_output

    @jit
    def loss(self, nn_params, x, y):
        resid = self.forward(nn_params, x) - output_scaler(y)
        resid = jnp.multiply(resid, jnp.array((1.0, 1.0, 1.0, 10.0, 10.0, 1.0)))
        return jnp.mean(jnp.square(resid))

    @jit
    def update(self, x, y, step_size):
      grads = grad(self.loss)(self.nn_parameters, x, y)
      return [(w - step_size * dw, b - step_size * db)
              for (w, b), (dw, db) in zip(self.nn_parameters, grads)]

    def accuracy(self, x, y):
        return self.loss(self.nn_parameters, x, y)

    def training_step(self, batch_generator, step_size):
        for x, y in batch_generator:
            self.nn_parameters = self.update(x, y, step_size)

    def train(self, batch_generator, step_size, num_epochs):
        for epoch in range(num_epochs):
            start_time = time.time()
            self.training_step(batch_generator, step_size)
            epoch_time = time.time() - start_time
            train_acc = self.accuracy(batch_generator._training_x, batch_generator._training_y)
            test_acc = self.accuracy(batch_generator._test_x, batch_generator._test_y)
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))

class Trainer:
    def __init__(self, num_epochs, batch_size, step_size):
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._step_size = step_size

class BatchGenerator:
    def __init__(self, batch_size, data):
        self._batch_size = batch_size
        self._training_x, self._training_y, self._test_x, self._test_y = get_datasets(data)
        self._data_size = self._training_x.shape[0]
        self._current_batch_index = 0
        self._max_batch_index = math.floor(int(self._data_size/self._batch_size))

    @staticmethod
    def get_datasets(data):
        n_rows = data.shape[0]
        jnp.random.shuffle(data)
        train_size = int(n_rows * 0.8)
        training, test = data[:train_size, :], data[train_size:, :]
        return training[:, :8], training[:, 8:], test[:, :8], test[:, 8:]

    def __iter__(self):
        self._current_batch_index = 0
        return self

    def __next__(self):
        if self._current_batch_index <= self._max_batch_index-1:
            start_index = self._current_batch_index * self._batch_size
            end_index = (self._current_batch_index+1) * self._batch_size
            result_x = self._training_x[start_index:end_index, :]
            result_y = self._training_y[start_index:end_index, :]
            self._current_batch_index += 1
            return jnp.array(result_x), jnp.array(result_y)
        else:
            raise StopIteration





