import numpy as np
from math import pi

E = 1e-10


def get_numerical_gradients(func: callable,
                            x: float,
                            y: float,
                            e: float) -> tuple:
    return (func(x+e, y) - func(x-e, y)) / (2 * e),  (func(x, y+e) - func(x, y-e)) / (2 * e)


def get_distance(x1: float,
                 y1: float,
                 x2: float,
                 y2: float) -> float:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_cyclical_lr(max_lr: float,
                    min_lr: float,
                    step: int,
                    cycle_size: int) -> float:
    return min_lr + (max_lr - min_lr) * np.sin(pi * float((step % cycle_size) / cycle_size))


def sgd_step(func: callable,
             x: float,
             y: float,
             lr: float) -> tuple:
    dx, dy = get_numerical_gradients(func, x, y, E)
    return x - lr * dx, y - lr * dy


def momentum_step(func: callable,
                  x: float,
                  y: float,
                  vx: float,
                  vy: float,
                  lr: float,
                  gamma: float) -> tuple:

    dx, dy = get_numerical_gradients(func, x, y, E)
    vx = gamma * vx + lr * dx
    vy = gamma * vy + lr * dy
    x = x - vx
    y = y - vy
    return x, y, vx, vy


def nesterov_step(func: callable,
                  x: float,
                  y: float,
                  vx: float,
                  vy: float,
                  lr: float,
                  gamma: float) -> tuple:

    dx, dy = get_numerical_gradients(func, x-gamma*vx, y-gamma*vy, E)
    vx = gamma * vx + lr * dx
    vy = gamma * vy + lr * dy
    x = x - vx
    y = y - vy
    return x, y, vx, vy


def adagrad_step(func: callable,
                 x: float,
                 y: float,
                 dx_history: list,
                 dy_history: list,
                 lr: float,
                 epsilon: float) -> tuple:
    dx, dy = get_numerical_gradients(func, x, y, E)
    x = x - (lr / np.sqrt((np.sum(np.array(dx_history + [dx])**2) + epsilon))) * dx
    y = y - (lr / np.sqrt((np.sum(np.array(dy_history + [dy])**2) + epsilon))) * dy
    return x, y, dx, dy


def adadelta_step(func: callable,
                  x: float,
                  y: float,
                  dx_ma: float,
                  dy_ma: float,
                  x_ma: float,
                  y_ma: float,
                  gamma: float,
                  lr: float,
                  epsilon: float) -> tuple:
    dx, dy = get_numerical_gradients(func, x, y, E)
    dx_ma = (1-gamma) * dx**2 + gamma * dx_ma
    dy_ma = (1-gamma) * dy**2 + gamma * dy_ma

    x = x - lr * (np.sqrt(x_ma + epsilon) / np.sqrt(dx_ma + epsilon)) * dx
    y = y - lr * (np.sqrt(y_ma + epsilon) / np.sqrt(dy_ma + epsilon)) * dy
    return x, y, dx, dy, dx_ma, dy_ma


def rmsprop_step(func: callable,
                 x: float,
                 y: float,
                 dx_ma: float,
                 dy_ma: float,
                 gamma: float,
                 lr: float,
                 epsilon: float) -> tuple:
    dx, dy = get_numerical_gradients(func, x, y, E)
    dx_ma = (1 - gamma) * dx**2 + gamma * dx_ma
    dy_ma = (1 - gamma) * dy**2 + gamma * dy_ma
    x = x - (lr / np.sqrt(dx_ma + epsilon))*dx
    y = y - (lr / np.sqrt(dy_ma + epsilon))*dy
    return x, y, dx, dy, dx_ma, dy_ma


def adam_step(func: callable,
              x: float,
              y:float,
              dx_ma: float,
              dy_ma: float,
              dx2_ma: float,
              dy2_ma: float,
              beta1: float,
              beta2: float,
              lr: float,
              epsilon: float,
              step: int
              ):
    dx, dy = get_numerical_gradients(func, x, y, E)

    dx_ma = (1 - beta1) * dx + beta1 * dx_ma
    dy_ma = (1 - beta1) * dy + beta1 * dy_ma

    dx2_ma = (1 - beta2) * dx ** 2 + beta2 * dx2_ma
    dy2_ma = (1 - beta2) * dy ** 2 + beta2 * dy2_ma

    dx_ma_hat = dx_ma / (1 - beta1 ** step)
    dy_ma_hat = dy_ma / (1 - beta1 ** step)

    dx2_ma_hat = dx2_ma / (1 - beta2 ** step)
    dy2_ma_hat = dy2_ma / (1 - beta2 ** step)

    x = x - (lr/(np.sqrt(dx2_ma_hat) + epsilon)) * dx_ma_hat
    y = y - (lr / (np.sqrt(dy2_ma_hat) + epsilon)) * dy_ma_hat
    return x, y, dx_ma, dy_ma, dx2_ma, dy2_ma
