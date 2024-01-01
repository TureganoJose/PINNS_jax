from main import dynamics
import h5py
import numpy as np
import numpy.typing as npt
from scipy.stats import qmc

import jax.numpy as jnp
from jax import vmap
# from jax import random
from jax.typing import ArrayLike
import jax
from multiprocessing import Pool
from multiprocessing import cpu_count

params = {}
params['m'] = jnp.float16(1480)  # [kg]
params['Iz'] = jnp.float16(1950)  # [kgm ^ 2]
params['a'] = jnp.float16(1.421)  # [m]
params['b'] = jnp.float16(1.029)  # [m]
params['mu'] = jnp.float16(1)  # [dimensionless]
params['g'] = jnp.float16(9.81)  # [m / s ^ 2]
params['dt'] = jnp.float16(0.01)  # sampling time

def dynamics_batch(x, u):
   # Pad control inputs: u
   # Padding is needed to differentiate as inputs should be the same size
   u = np.pad(u, ((0,0),(0, 4)), 'constant')
   print(u.shape)
   dynamics_map = vmap(dynamics, in_axes=[0, 0, None])(x, u, params)
   return dynamics_map

if __name__ == '__main__':
    n_devices = jax.local_device_count()
    print(n_devices)
    print(cpu_count())

    sampler = qmc.LatinHypercube(d=5)
    sample = sampler.random(n=7800000)
    # [x y yaw vx vy yaw_rate steering Fx]
    #          [vx   vy   yaw_rate steering Fx]
    l_bounds = [0.0, -5.0, -0.35, -0.26,   -1000]
    u_bounds = [20.0, 5.0,  0.35,  0.26,    1000]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    # print(type(sample_scaled))
    # print(sample_scaled.shape)
    small_sample = sample_scaled[0:7800000]
    training_data = np.empty([0, 6+2+6], float)

    x = np.concatenate((np.zeros((7800000, 3)), small_sample[:, 0:3]), axis=1)
    u = small_sample[:, 3:]
    y = dynamics_batch(x, u)
    training_data = np.concatenate((x, u, y), axis=1)
    training_data = training_data[ np.abs(training_data[:,12])<3,:]
    training_data = training_data[np.abs(training_data[:,13]) < 0.26, :]
    # for input in small_sample:
    #     x = np.concatenate(([0.0, 0.0, 0.0], input[0:3]))
    #     u = input[3:]
    #     y = dynamics(x, u, params)
    #     if abs(y[4]) > 3 or abs(y[5])>0.26:
    #         continue
    #     new_row = np.concatenate((x, u, y))
    #     training_data = np.vstack((training_data, new_row))
        # print(x)
        # print(u)
        # print(y)
        # print(training_data)
    print(training_data.shape)
    h5f = h5py.File('data_curated_7_8m.h5', 'w')
    h5f.create_dataset('dataset_1', data=training_data)

    h5f.close()

    # h5f = h5py.File('data_curated_500k.h5', 'r')
    # b = h5f['dataset_1'][:]
    # h5f.close()
    # print(b.shape)
    # np.allclose(training_data, b)
    True