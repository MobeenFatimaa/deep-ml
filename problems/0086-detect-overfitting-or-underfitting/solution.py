def model_fit_quality(training_accuracy, test_accuracy):
    """
    Determine if the model is overfitting, underfitting, or a good fit based on training and test accuracy.
    :param training_accuracy: float, training accuracy of the model (0 <= training_accuracy <= 1)
    :param test_accuracy: float, test accuracy of the model (0 <= test_accuracy <= 1)
    :return: int, one of 1, -1, or 0.
    """
    # Overfitting: Training accuracy is significantly higher than test accuracy (> 0.2 difference)
    if (training_accuracy - test_accuracy) > 0.2:
        return 1
    
    # Underfitting: Both training and test accuracies are below 0.7
    elif training_accuracy < 0.7 and test_accuracy < 0.7:
        return -1
    
    # Good fit: Neither condition is met
    else:
        return 0