class StepLRScheduler:
    def __init__(self, initial_lr: float, step_size: int, gamma: float):
        """
        Initializes the StepLR learning rate scheduler.
        
        Args:
            initial_lr (float): The starting learning rate.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
        """
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        """
        Returns the learning rate for a given epoch.
        
        Args:
            epoch (int): The current epoch number.
            
        Returns:
            float: The learning rate rounded to 4 decimal places.
        """
        # Determine how many times the learning rate has stepped down
        num_steps = epoch // self.step_size
        
        # Calculate current learning rate
        lr = self.initial_lr * (self.gamma ** num_steps)
        
        return round(float(lr), 4)