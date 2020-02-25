from schedulers.scheduler import Scheduler


class VanillaScheduler(Scheduler):
    def __init__(self, init_lr: float) -> None:
        super(VanillaScheduler, self).__init__(init_lr)
        self.init_lr = init_lr
        self.step = 1

    def get_lr(self) -> float:
        return self.init_lr
