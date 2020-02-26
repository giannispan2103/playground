from schedulers.scheduler import Scheduler
from utils import get_cyclical_lr


class CLRScheduler(Scheduler):
    def __init__(self,
                 init_lr: float,
                 max_lr: float,
                 cycle_size: int):
        super(CLRScheduler, self).__init__(init_lr)
        self.cycle_size = cycle_size
        self.max_lr = max_lr

    def get_lr(self) -> float:
        return get_cyclical_lr(self.max_lr, self.init_lr, self.step, self.cycle_size)
