from optimizers.optimizer import Optimizer
from utils import sgd_step

class SGD(Optimizer):
    def __init__(self, iterations: int, init_x: float, init_y: float, func: callable, lr: float) -> None:
        super(SGD, self).__init__(iterations, init_x, init_y, func, lr)

    def step(self) -> None:
        self.x, self.y = sgd_step(self.func, self.x, self.y)
        self.x_values.append(self.x)
        self.y_values.append(self.y)
        loss = self.func(self.x, self.y)
        self.f_values.append(loss)