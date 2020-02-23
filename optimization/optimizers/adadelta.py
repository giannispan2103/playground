from optimizers.optimizer import Optimizer
from utils import adadelta_step


class Adadelta(Optimizer):
    def __init__(self,
                 iterations: int,
                 init_x: float,
                 init_y: float,
                 func: callable,
                 gamma: float,
                 epsilon: float,
                 lr: float = 1.0) -> None:
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
        self.x, self.y, self.dx, self.dy, self.dx_ma, self.dy_ma = adadelta_step(func=self.func,
                                                                                 x=self.x,
                                                                                 y=self.y,
                                                                                 dx_ma=self.dx_ma,
                                                                                 dy_ma=self.dy_ma,
                                                                                 x_ma=self.x_ma,
                                                                                 y_ma=self.y_ma,
                                                                                 epsilon=self.epsilon,
                                                                                 gamma=self.gamma,
                                                                                 lr=self.lr)

        loss = self.func(self.x, self.y)
        self.f_values.append(loss)
        self.x_ma = (1 - self.gamma) * (self.x - self.x_values[-1]) ** 2 + self.gamma * self.x_ma
        self.y_ma = (1 - self.gamma) * (self.y - self.y_values[-1]) ** 2 + self.gamma * self.y_ma
        self.x_values.append(self.x)
        self.y_values.append(self.y)

