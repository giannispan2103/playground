from optimizers.optimizer import Optimizer
from utils import adagrad_step
from schedulers.scheduler import Scheduler


class Adagrad(Optimizer):
    def __init__(self,
                 iterations: int,
                 init_x: float,
                 init_y: float,
                 func: callable,
                 scheduler: Scheduler,
                 epsilon: float) -> None:
        super(Adagrad, self).__init__(iterations=iterations,
                                      init_x=init_x,
                                      init_y=init_y,
                                      func=func,
                                      scheduler=scheduler)
        self.epsilon = epsilon
        self.dx = 0.0
        self.dy = 0.0
        self.dx_history = []
        self.dy_history = []

    def step(self) -> None:
        self.x, self.y, self.dx, self.dy = adagrad_step(func=self.func, x=self.x, y=self.y,
                                                        dx_history=self.dx_history,
                                                        dy_history=self.dy_history,
                                                        epsilon=self.epsilon, lr=self.scheduler.get_lr())

        self.x_values.append(self.x)
        self.y_values.append(self.y)
        loss = self.func(self.x, self.y)
        self.f_values.append(loss)
        self.dx_history.append(self.dx)
        self.dy_history.append(self.dy)