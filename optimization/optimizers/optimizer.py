from utils import get_distance
from schedulers.scheduler import Scheduler


class Optimizer(object):
    def __init__(self, iterations: int,
                 init_x: float,
                 init_y: float,
                 func: callable,
                 scheduler: Scheduler) -> None:
        self.iterations = iterations
        self.func = func
        self.f_values = [self.func(init_x, init_y)]
        self.x_values = [init_x]
        self.y_values = [init_y]
        self.scheduler = scheduler
        self.x = init_x
        self.y = init_y

    def step(self) -> None:
        raise NotImplementedError()

    def simulate(self) -> None:
        for it in range(1, self.iterations + 1):
            self.step()
            self.scheduler.update()

    def distance(self,
                 target_x: float,
                 target_y: float) -> float:
        return get_distance(x1=self.x,
                            y1=self.y,
                            x2=target_x,
                            y2=target_y)
