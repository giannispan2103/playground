from optimizers.optimizer import Optimizer
from utils import adam_step
from schedulers.scheduler import Scheduler


class Adam(Optimizer):
    def __init__(self,
                 iterations: int,
                 init_x: float,
                 init_y: float,
                 func: callable,
                 scheduler: Scheduler,
                 beta1: float,
                 beta2: float,
                 epsilon: float) -> None:
        super(Adam, self).__init__(iterations=iterations,
                                       init_x=init_x,
                                       init_y=init_y,
                                       func=func,
                                       scheduler=scheduler)

        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.dx_ma = 0.0
        self.dy_ma = 0.0
        self.dx2_ma = 0.0
        self.dy2_ma = 0.0

    def step(self) -> None:
        self.x, self.y,  self.dx_ma, self.dy_ma, self.dx2_ma, self.dy2_ma = adam_step(func=self.func,
                                                                                      x=self.x,
                                                                                      y=self.y,
                                                                                      dx_ma=self.dx_ma,
                                                                                      dy_ma=self.dy_ma,
                                                                                      dx2_ma=self.dx2_ma,
                                                                                      dy2_ma=self.dy2_ma,
                                                                                      epsilon=self.epsilon,
                                                                                      beta1=self.beta1,
                                                                                      beta2=self.beta2,
                                                                                      lr=self.scheduler.get_lr(),
                                                                                      step=self.scheduler.step)

        loss = self.func(self.x, self.y)
        self.f_values.append(loss)
        self.x_values.append(self.x)
        self.y_values.append(self.y)