from schedulers.scheduler import Scheduler


class ExpDecayScheduler(Scheduler):
    def __init__(self,
                 init_lr: float,
                 decay_size: float,
                 cycle_size: int):
        super(ExpDecayScheduler, self).__init__(init_lr)
        self.cycle_size = cycle_size
        self.decay_size = decay_size

    def get_lr(self) -> float:
        exponent = int(self.step / self.cycle_size)
        return self.init_lr * (1 - self.decay_size) ** exponent

