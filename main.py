import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import vmap, grad
# from jax import random
from jax.typing import ArrayLike

nDArrayFloat = npt.NDArray[np.float_]

def dynamics_residuals(dx_dt: ArrayLike, x: ArrayLike, u: ArrayLike, params: dict) -> ArrayLike:
    # Function
    # Inputs: x, dx_dt, u, params
    # Returns: 0.0

    m = params['m']  # [kg]
    Iz = params['Iz']  # [kgm ^ 2]
    a = params['a']  # [m]
    b = params['b']  # [m]
    mu = params['mu']  # [dimensionless]
    g = params['g']  # [m / s ^ 2]

    # definition of slip angles and forces
    beta_f = u[0] - (x[4] + a * x[5]) / (x[3])
    beta_r = -(x[4] - b * x[5]) / (x[3])

    Fz_f = (m * g * b) / (a + b)
    Fz_r = (m * g * a) / (a + b)

    Fy_f = mu * Fz_f * beta_f
    Fy_r = mu * Fz_r * beta_r

    return jnp.asarray((
        ((dx_dt[0] - x[5] * x[4])) - (1/m)*(u[1] * jnp.cos(u[0]) - Fy_f * jnp.sin(u[0])),
        ((dx_dt[1] + x[5] * x[3])) - (1/m)*(u[1] * jnp.sin(u[0]) + Fy_f * jnp.cos(u[0]) + Fy_r - m * x[5] * x[3]),
        (dx_dt[2]) - (1/Iz)*((u[1] * jnp.sin(u[0]) + Fy_f * jnp.cos(u[0])) * a - Fy_r * b)))

def dynamics_residuals_batch(dx_dt, x, u, params):
    # Pad control inputs: u
    u = jnp.pad(u, ((0, 0), (0, 4)), 'constant')
    dx_dt = jnp.pad(dx_dt, ((0, 0), (0, 3)), 'constant')
    dynamics_residuals_map = vmap(dynamics_residuals, in_axes=[0, 0, 0, None])(dx_dt, x, u, params)
    return dynamics_residuals_map

def dynamics(x: ArrayLike, u: ArrayLike, params: dict) -> ArrayLike:
    # Function
    # Inputs: x_t, u_t, params
    # Returns: x_t + 1

    # function[x_plus, df_x, df_u, df_xx, df_uu, df_ux, pfxx, pfuu, pfux] = dynamics_ale(xx, uu, params, pp)

    m = params['m']  # [kg]
    Iz = params['Iz']  # [kgm ^ 2]
    a = params['a']  # [m]
    b = params['b']  # [m]
    mu = params['mu']  # [dimensionless]
    g = params['g']  # [m / s ^ 2]
    dt = params['dt']  # sampling time

    # definition of slip angles and forces
    beta_f = u[0] - (x[4] + a * x[5]) / (x[3])
    beta_r = -(x[4] - b * x[5]) / (x[3])

    Fz_f = (m * g * b) / (a + b)
    Fz_r = (m * g * a) / (a + b)

    Fy_f = mu * Fz_f * beta_f
    Fy_r = mu * Fz_r * beta_r

    return jnp.asarray((
        x[0] + dt * (x[3] * jnp.cos(x[2]) - x[4] * jnp.sin(x[2])),
        x[1] + dt * (x[3] * jnp.sin(x[2]) + x[4] * jnp.cos(x[2])),
        x[2] + dt * (x[5]),
        x[3] + dt * (1 / m) * (u[1] * jnp.cos(u[0]) - Fy_f * jnp.sin(u[0]) + m * x[5] * x[4]),
        x[4] + dt * (1 / m) * (u[1] * jnp.sin(u[0]) + Fy_f * jnp.cos(u[0]) + Fy_r - m * x[5] * x[3]),
        x[5] + dt * (1 / Iz) * ((u[1] * jnp.sin(u[0]) + Fy_f * jnp.cos(u[0])) * a - Fy_r * b)))


if __name__ == '__main__':
    x = jnp.zeros((6))
    u = jnp.ones((2))

    params = {}
    params['m'] = jnp.float16(1480)  # [kg]
    params['Iz'] = jnp.float16(1950)  # [kgm ^ 2]
    params['a'] = jnp.float16(1.421)  # [m]
    params['b'] = jnp.float16(1.029)  # [m]
    params['mu'] = jnp.float16(1)  # [dimensionless]
    params['g'] = jnp.float16(9.81)  # [m / s ^ 2]
    params['dt'] = jnp.float16(0.01)  # sampling time

    x = jnp.array((0.0, 0.0, 0.0, 6.5, 1.1, 0.1))
    u = jnp.array((0.1, 900.0))
    y = dynamics(x, u, params)
    dx_dt = (y[3:6]-x[3:6])/params['dt']
    resid_physics=dynamics_residuals(dx_dt, x, u, params)
    resid_physics= dynamics_residuals_batch(np.tile(dx_dt, (128,1)), np.tile(x, (128,1)), np.tile(u,(128,1)),params)
    # dfn_dx = grad(dynamics, argnums=0)(x, u, params)
    # dfn_du = grad(dynamics, argnums=1)(x, u, params)
    # dfn_dparams = grad(dynamics, argnums=2)(x, u, params)
    x_small = jnp.asarray((1.001, 2.001, 1.001, 1.001, 1.001, 1.001))
    J1 = jax.jacfwd(dynamics, argnums=0)(x_small, u, params)
    print(type(J1))
    print(len(J1))
    print(J1)
