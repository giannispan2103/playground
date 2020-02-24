from optimizers.optimizer import Optimizer
from utils import sgd_step
from schedulers.scheduler import Scheduler


class SGD(Optimizer):
    def __init__(self,
                 iterations: int,
                 init_x: float,
                 init_y: float,
                 func: callable,
                 scheduler: Scheduler) -> None:
        super(SGD, self).__init__(iterations, init_x, init_y, func, scheduler)

    def step(self) -> None:
        self.x, self.y = sgd_step(self.func, self.x, self.y, lr=self.scheduler.get_lr())
        self.x_values.append(self.x)
        self.y_values.append(self.y)
        loss = self.func(self.x, self.y)
        self.f_values.append(loss)
