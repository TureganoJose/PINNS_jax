from main import dynamics
from vd_neural_network import BatchGenerator, VehicleDynamicsNN, output_descalar

import h5py
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp  # YOU should jnp by default

params = {}
params['m'] = jnp.float16(1480)  # [kg]
params['Iz'] = jnp.float16(1950)  # [kgm ^ 2]
params['a'] = jnp.float16(1.421)  # [m]
params['b'] = jnp.float16(1.029)  # [m]
params['mu'] = jnp.float16(1)  # [dimensionless]
params['g'] = jnp.float16(9.81)  # [m / s ^ 2]
params['dt'] = jnp.float16(0.01)  # sampling time

if __name__=="__main__":
    h5f = h5py.File('data_curated_7_8m.h5', 'r')
    data = h5f['dataset_1'][:]
    h5f.close()

    batched_data = BatchGenerator(batch_size=128, data=data)
    data_u_bound = data.max(axis=0)
    data_l_bound = data.min(axis=0)
    print(data_u_bound)
    print(data_l_bound)
    print(batched_data._training_x.shape)

    model = VehicleDynamicsNN(layer_sizes=[8, 128, 256, 128, 8, 6])
    model.train(batched_data, step_size=0.005, num_epochs=5)

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
        y_nn = output_descalar(model.predict(np.concatenate((x,u))))
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