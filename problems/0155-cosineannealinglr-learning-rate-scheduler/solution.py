import math

class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr, T_max, min_lr):
        """
        Initializes the Cosine Annealing LR scheduler.
        :param initial_lr: Initial learning rate (float)
        :param T_max: Maximum number of iterations/epochs for half-cycle (int)
        :param min_lr: Minimum learning rate (float)
        """
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.min_lr = min_lr

    def get_lr(self, epoch):
        """
        Calculates and returns the learning rate for a given epoch.
        :param epoch: Current epoch/iteration (int)
        :return: Learning rate rounded to 4 decimal places (float)
        """
        cycle = 2 * self.T_max
        e = epoch % cycle
        
        # Cosine annealing formula
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + math.cos(math.pi * e / self.T_max))
        
        return round(float(lr), 4)