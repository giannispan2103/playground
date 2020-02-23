from utils import get_distance


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