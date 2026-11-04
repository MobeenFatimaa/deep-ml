class ExponentialLRScheduler:
    def __init__(self, initial_lr: float, gamma: float):
        """
        Initializes the ExponentialLR learning rate scheduler.
        
        Args:
            initial_lr (float): The starting learning rate.
            gamma (float): Multiplicative factor of learning rate decay per epoch.
        """
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        """
        Returns the learning rate for a given epoch.
        
        Args:
            epoch (int): The current epoch number.
            
        Returns:
            float: The learning rate rounded to 4 decimal places.
        """
        # Calculate exponential decay: initial_lr * (gamma ^ epoch)
        lr = self.initial_lr * (self.gamma ** epoch)
        
        return round(float(lr), 4)