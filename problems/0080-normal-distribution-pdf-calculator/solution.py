import math

def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated.
    :param mean: The mean (μ) of the distribution.
    :param std_dev: The standard deviation (σ) of the distribution.
    """
    # Prevent division by zero or invalid standard deviation
    if std_dev <= 0:
        raise ValueError("Standard deviation must be greater than zero.")
    
    # Formula for Normal Distribution PDF:
    # f(x) = (1 / (std_dev * sqrt(2 * pi))) * exp(-0.5 * ((x - mean) / std_dev) ** 2)
    coefficient = 1 / (std_dev * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mean) / std_dev) ** 2
    val = coefficient * math.exp(exponent)
    
    return round(val, 5)