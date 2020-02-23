from utils import sgd_step, momentum_step, adagrad_step, adadelta_step, rmsprop_step, get_distance


class Optimizer(object):
    def __init__(self, iterations: int, init_x: float, init_y: float, func: callable, lr: float) -> None:
        self.init_x = init_x
        self.init_y = init_y
        self.iterations = iterations
        self.func = func
        self.f_values = [self.func(self.init_x, self.init_y)]
        self.x_values = [init_x]
        self.y_values = [init_y]
        self.lr = lr
        self.x = init_x
        self.y = init_y
        self.current_iteration = 0

    def step(self) -> None:
        raise NotImplementedError

    def simulate(self) -> None:
        for it in range(1, self.iterations + 1):
            self.step()
            self.current_iteration = it

    def distance(self, target_x: float, target_y: float) -> float:
        return get_distance(self.x, self.y, target_x, target_y)

    def decay_policy(self) -> None:
        pass

    def cyclical_decay(self) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, iterations: int, init_x: float, init_y: float, func: callable, lr: float) -> None:
        super(SGD, self).__init__(iterations, init_x, init_y, func, lr)

    def step(self) -> None:
        self.x, self.y = sgd_step(self.func, self.x, self.y)
        self.x_values.append(self.x)
        self.y_values.append(self.y)
        loss = self.func(self.x, self.y)
        self.f_values.append(loss)


class Momentum(Optimizer):
    def __init__(self, iterations: int,
                 init_x: float, init_y: float,
                 func: callable,
                 lr: float, gamma: float = 0.9, nesterov: bool = False) -> None:
        super(Momentum, self).__init__(iterations=iterations, init_x=init_x, init_y=init_y, func=func, lr=lr)
        self.gamma = gamma
        self.vx = 0.0
        self.vy = 0.0
        self.vx_history = []
        self.vy_history = []
        self.nesterov = nesterov

    def step(self) -> None:
        self.x, self.y, self.vx, self.vy = momentum_step(func=self.func, x=self.x, y=self.y, vx=self.vx,
                                                         vy=self.vy, gamma=self.gamma, nesterov=self.nesterov)
        self.x_values.append(self.x)
        self.y_values.append(self.y)
        loss = self.func(self.x, self.y)
        self.f_values.append(loss)
        self.vx_history.append(self.vx)
        self.vy_history.append(self.vy)


class Adagrad(Optimizer):
    def __init__(self, iterations: int,
                 init_x: float, init_y: float,
                 func: callable,
                 lr: float, epsilon: float = 0.00000001) -> None:
        super(Adagrad, self).__init__(iterations=iterations, init_x=init_x, init_y=init_y, func=func, lr=lr)
        self.epsilon = epsilon
        self.dx = 0.0
        self.dy = 0.0
        self.dx_history = []
        self.dy_history = []

    def step(self) -> None:
        self.x, self.y, self.dx, self.dy = adagrad_step(func=self.func, x=self.x, y=self.y,
                                                        dx_history=self.dx_history,
                                                        dy_history=self.dy_history,
                                                        epsilon=self.epsilon)

        self.x_values.append(self.x)
        self.y_values.append(self.y)
        loss = self.func(self.x, self.y)
        self.f_values.append(loss)
        self.dx_history.append(self.dx)
        self.dy_history.append(self.dy)


class Adadelta(Optimizer):
    def __init__(self, iterations: int,
                 init_x: float, init_y: float,
                 func: callable,
                 gamma: float,
                 lr: float = 1.0,
                 epsilon: float = 0.00000001) -> None:
        super(Adadelta, self).__init__(iterations=iterations, init_x=init_x, init_y=init_y, func=func, lr=lr)
        self.epsilon = epsilon
        self.gamma = gamma
        self.dx = 0.0
        self.dy = 0.0
        self.dx_ma = 0.0
        self.dy_ma = 0.0
        self.x_ma = init_x
        self.y_ma = init_y

    def step(self) -> None:
        self.x, self.y, self.dx, self.dy, self.dx_ma, self.dy_ma = adadelta_step(func=self.func, x=self.x, y=self.y,
                                                                                 dx_ma=self.dx_ma,
                                                                                 dy_ma=self.dy_ma,
                                                                                 x_ma=self.x_ma,
                                                                                 y_ma=self.y_ma,
                                                                                 epsilon=self.epsilon, gamma=self.gamma,
                                                                                 lr=self.lr)

        loss = self.func(self.x, self.y)
        self.f_values.append(loss)
        self.x_ma = (1 - self.gamma) * (self.x - self.x_values[-1]) ** 2 + self.gamma * self.x_ma
        self.y_ma = (1 - self.gamma) * (self.y - self.y_values[-1]) ** 2 + self.gamma * self.y_ma
        self.x_values.append(self.x)
        self.y_values.append(self.y)


class RMSProp(Optimizer):
    def __init__(self, iterations: int,
                 init_x: float, init_y: float,
                 func: callable,
                 gamma: float,
                 lr: float,
                 epsilon: float = 0.00000001) -> None:
        super(RMSProp, self).__init__(iterations=iterations, init_x=init_x, init_y=init_y, func=func, lr=lr)
        self.epsilon = epsilon
        self.gamma = gamma
        self.dx = 0.0
        self.dy = 0.0
        self.dx_ma = 0.0
        self.dy_ma = 0.0

    def step(self) -> None:
        self.x, self.y, self.dx, self.dy, self.dx_ma, self.dy_ma = rmsprop_step(func=self.func, x=self.x, y=self.y,
                                                                                dx_ma=self.dx_ma,
                                                                                dy_ma=self.dy_ma,
                                                                                lr=self.lr,
                                                                                epsilon=self.epsilon, gamma=self.gamma)
        self.x_values.append(self.x)
        self.y_values.append(self.y)
        loss = self.func(self.x, self.y)
        self.f_values.append(loss)