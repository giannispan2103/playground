from optimizers.optimizer import Optimizer
from utils import momentum_step
from schedulers.scheduler import Scheduler


class Momentum(Optimizer):
    def __init__(self,
                 iterations: int,
                 init_x: float,
                 init_y: float,
                 func: callable,
                 scheduler: Scheduler,
                 gamma: float) -> None:
        super(Momentum, self).__init__(iterations=iterations,
                                       init_x=init_x,
                                       init_y=init_y,
                                       func=func,
                                       scheduler=scheduler)
        self.gamma = gamma
        self.vx = 0.0
        self.vy = 0.0
        self.vx_history = []
        self.vy_history = []

    def step(self) -> None:
        self.x, self.y, self.vx, self.vy = momentum_step(func=self.func,
                                                         x=self.x,
                                                         y=self.y,
                                                         vx=self.vx,
                                                         vy=self.vy,
                                                         gamma=self.gamma,
                                                         lr=self.scheduler.get_lr())
        self.x_values.append(self.x)
        self.y_values.append(self.y)
        loss = self.func(self.x, self.y)
        self.f_values.append(loss)
        self.vx_history.append(self.vx)
        self.vy_history.append(self.vy)