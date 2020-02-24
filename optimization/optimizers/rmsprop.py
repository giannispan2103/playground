from optimizers.optimizer import Optimizer
from utils import rmsprop_step
from schedulers.scheduler import Scheduler

class RMSProp(Optimizer):
    def __init__(self,
                 iterations: int,
                 init_x: float,
                 init_y: float,
                 func: callable,
                 gamma: float,
                 scheduler: Scheduler,
                 epsilon: float) -> None:
        super(RMSProp, self).__init__(iterations=iterations, init_x=init_x, init_y=init_y, func=func,
                                      scheduler=scheduler)
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
                                                                                lr=self.scheduler.get_lr(),
                                                                                epsilon=self.epsilon, gamma=self.gamma)
        self.x_values.append(self.x)
        self.y_values.append(self.y)
        loss = self.func(self.x, self.y)
        self.f_values.append(loss)



