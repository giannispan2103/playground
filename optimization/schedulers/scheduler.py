class Scheduler(object):
    def __init__(self,
                 init_lr: float) -> None:
        self.init_lr = init_lr
        self.step = 1

    def get_lr(self) -> float:
        raise NotImplementedError()

    def update(self) -> None:
        self.step += 1
