import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jax import random, nn
from main import dynamics, dynamics_residuals

import h5py
import numpy as np
import time
import math
import copy
import matplotlib.pyplot as plt

# Constants
# [x y yaw vx vy yaw_rate steering Fx]
l_bounds = jnp.array((-0.5, -0.5, -0.5, 0.0, -5.0, -0.35, -0.26, -1000))
u_bounds = jnp.array((0.5, 0.5, 0.5, 20.0, 5.0, 0.35, 0.26, 1000))

# [x y yaw vx vy yaw_rate]
l_out_bounds = jnp.array((-0.5, -0.5, -0.5, 0.0, -5.0, -0.35))
u_out_bounds = jnp.array((0.5, 0.5, 0.5, 20.0, 5.0, 0.35))

layer_sizes = [8, 128, 256, 128, 8, 6]
step_size = 0.005
num_epochs = 3
batch_size = 128
n_targets = 6

params = {}
params['m'] = jnp.float16(1480)  # [kg]
params['Iz'] = jnp.float16(1950)  # [kgm ^ 2]
params['a'] = jnp.float16(1.421)  # [m]
params['b'] = jnp.float16(1.029)  # [m]
params['mu'] = jnp.float16(1)  # [dimensionless]
params['g'] = jnp.float16(9.81)  # [m / s ^ 2]
params['dt'] = jnp.float16(0.01)  # sampling time

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


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

def predict(nn_params, x):
    x_scaled = input_scaler(x)
    x_scaled_ori = copy.deepcopy(x_scaled)
    # per-example predictions
    for w, b in nn_params[:-1]:
        outputs = jnp.dot(w, x_scaled) + b
        x_scaled = nn.relu(outputs)
    final_w, final_b = nn_params[-1]
    # Residuals
    x_scaled = x_scaled_ori + x_scaled
    final_output = jnp.dot(final_w, x_scaled) + final_b
    return final_output

def accuracy(nn_parameters, x, y):
  return loss_with_regularization(nn_parameters, x, y)

# def loss(nn_parameters, x_batch, targets):
#   preds = batched_predict(nn_parameters, x_batch)
#   return -jnp.mean(preds * targets)

def dynamics_batch(x, u):
   # Pad control inputs: u
   u = np.pad(u, ((0,0),(0, 4)), 'constant')
   dynamics_map = vmap(dynamics, in_axes=[0, 0, None])(x, u, params)
   return dynamics_map

def dynamics_residuals_batch(dx_dt, x, u):
    # Pad control inputs: u
    u = jnp.pad(u, ((0, 0), (0, 4)), 'constant')
    dx_dt = jnp.pad(dx_dt, ((0, 0), (0, 3)), 'constant')
    dynamics_residuals_map = vmap(dynamics_residuals, in_axes=[0, 0, 0, None])(dx_dt, x, u, params)
    return dynamics_residuals_map

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

#@jit
def loss(nn_parameters, x, y):
    resid = batched_predict(nn_parameters, x) - output_scaler(y)
    resid = jnp.multiply(resid, jnp.array((1.0, 1.0, 1.0, 10.0, 10.0, 1.0)))
    return jnp.mean(jnp.square(resid))


# Physics
def loss_with_regularization(nn_parameters, x, y):
    y_pred_scaled = batched_predict(nn_parameters, x)
    data_resid = y_pred_scaled - output_scaler(y)
    data_resid = jnp.multiply(data_resid, jnp.array((1.0, 1.0, 1.0, 10.0, 10.0, 1.0)))
    y_pred = output_descalar(y_pred_scaled)
    dx_dt = jnp.multiply(y_pred[:,3:6]-x[:, 3:6], 1/0.01)
    physics_resid = 0.001 * dynamics_residuals_batch(dx_dt, x[:, :6], x[:, 6:])
    # jax.debug.print("resid {x}", x=physics_resid)
    # jax.debug.print("dx_dt {x}", x=dx_dt)
    return jnp.mean(jnp.square(data_resid)) + jnp.mean(jnp.square(physics_resid))

@jit
def update(nn_parameters, x, y):
  grads = grad(loss_with_regularization)(nn_parameters, x, y)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(nn_parameters, grads)]

def get_datasets(data):
    n_rows = data.shape[0]
    np.random.shuffle(data)
    train_size = int(n_rows * 0.8)
    training, test = data[:train_size, :], data[train_size:, :]
    return training[:, :8], training[:, 8:], test[:, :8], test[:, 8:]

def get_train_batches(epoch, data_x, data_y):
    return data_x[epoch*batch_size:(1+epoch)*batch_size, :],data_y[epoch*batch_size:(1+epoch)*batch_size, :]

class BatchGenerator:
    def __init__(self, batch_size, data_x, data_y):
        self._batch_size = batch_size
        self._data_x = data_x
        self._data_y = data_y
        self._data_size = data_x.shape[0]
        self._current_batch_index = 0
        self._max_batch_index = math.floor(int(self._data_size/self._batch_size))

    def __iter__(self):
        self._current_batch_index = 0
        return self

    def __next__(self):
        if self._current_batch_index <= self._max_batch_index-1:
            start_index = self._current_batch_index * self._batch_size
            end_index = (self._current_batch_index+1) * self._batch_size
            result_x = self._data_x[start_index:end_index, :]
            result_y = self._data_y[start_index:end_index, :]
            self._current_batch_index += 1
            return result_x, result_y
        else:
            raise StopIteration


if __name__=='__main__':
    h5f = h5py.File('data_curated_7_8m.h5', 'r')
    training_data = h5f['dataset_1'][:]
    data_u_bound = training_data.max(axis=0)
    data_l_bound = training_data.min(axis=0)
    print(data_u_bound)
    print(data_l_bound)

    # plt.hist(training_data[:, 6], bins=100)
    # plt.show()
    # l_bounds = np.concatenate((np.array((-0.5, -0.5, -0.5)), data_l_bound[3:8]))
    # u_bounds = np.concatenate((np.array(( 0.5, 0.5,   0.5)), data_u_bound[3:8]))

    h5f.close()
    print(training_data.shape)

    training_x, training_y, test_x, test_y = get_datasets(training_data)

    print((training_x.shape))
    print((training_y.shape))
    print((test_x.shape))
    print((test_y.shape))

    x = training_x[:batch_size,:6]
    u = training_x[:batch_size, 6:]
    testing_batch = dynamics_batch(x, u)
    print(input_scaler(training_x[0, :]))
    nn_parameters = init_network_params(layer_sizes, random.PRNGKey(0))
    print(len(nn_parameters))

    for epoch in range(num_epochs):
        start_time = time.time()
        # Not really epoch
        for x, y in BatchGenerator(batch_size, training_x, training_y):
            # x, y = get_train_batches(epoch, training_x, training_y)
            nn_parameters = update(nn_parameters, x, y)
        epoch_time = time.time() - start_time
        train_acc = accuracy(nn_parameters, training_x, training_y)
        test_acc = accuracy(nn_parameters, test_x, test_y)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
    # [x y yaw vx vy yaw_rate steering Fx]
    x = np.array((0.0, 0.0, 0.0, 3.5, 0.0, 0.00))
    u = np.array((0.1, 750))
    y_real = dynamics(x, u, params)
    y_nn = output_descalar(predict(nn_parameters, np.concatenate((x,u))))
    print("real output:{}".format(y_real))
    print("nn   output:{}".format(y_nn))


    x = np.array((0.0, 0.0, 0.0, 3.5, 0.0, 0.0))
    n_sim_steps = 100
    time = np.zeros((n_sim_steps,))
    vx_real = np.zeros((n_sim_steps,))
    vx_nn = np.zeros((n_sim_steps,))
    vy_real = np.zeros((n_sim_steps,))
    vy_nn = np.zeros((n_sim_steps,))
    yaw_rate_real = np.zeros((n_sim_steps,))
    yaw_rate_nn = np.zeros((n_sim_steps,))
    for i in range(n_sim_steps):
        u = np.array((-i*0.004, -300))
        y_real = dynamics(x, u, params)
        y_nn = output_descalar(predict(nn_parameters, np.concatenate((x,u))))
        time[i] = i*0.01
        vx_real[i] = y_real[3]
        vx_nn[i] = y_nn[3]
        vy_real[i] = y_real[4]
        vy_nn[i] = y_nn[4]
        yaw_rate_real[i] = y_real[5]
        yaw_rate_nn[i] = y_nn[5]
        x = jnp.concatenate((jnp.array((0.0, 0.0, 0.0)), y_real[3:]))

    # Create two subplots and unpack the output array immediately
    plt.subplot(1, 3, 1)
    plt.plot(time, vx_real, label='real')
    plt.plot(time, vx_nn, label='nn')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(time, vy_real, label='real')
    plt.plot(time, vy_nn, label='nn')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(time, yaw_rate_real, label='real')
    plt.plot(time, yaw_rate_nn, label='nn')
    plt.legend()
    plt.show()

    # `batched_predict` has the same call signature as `predict`
    # batched_preds = batched_predict(nn_parameters, random_flattened_images)
    # print(batched_preds.shape)