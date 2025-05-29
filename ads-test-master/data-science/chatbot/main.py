def describe_model():
    """
    Describe the model type, shape, layers, etc.
    """
    raise NotImplementedError()


def report_model_performance():
    """
    Report the model's performance on the training and test datasets.
    """
    raise NotImplementedError()


def interact_with_model():
    """
    Accept user input from the console, in the form of a string, and output the model's predictions and its confidence score.
    Any extra data that you want to output, please do so.
    """
    raise NotImplementedError()


if __name__ == '__main__':
    describe_model()
    report_model_performance()
    interact_with_model()
