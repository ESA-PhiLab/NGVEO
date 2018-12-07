import numpy as np
from dlt.basic.pytorch_utils import torch_to_np

def balanced_accuracy(prediction, target):
    prediction =  torch_to_np(prediction)
    target = torch_to_np(target)

    #Convert prdiction score to classes
    prediction =_softmax(prediction,1)
    prediction = np.argmax(prediction,1)

    #Remove ignore pixels
    target = target[target != -100]
    prediction = prediction[target!=-100]

    class_accuracies = []
    for cls in np.unique(target):
        acc_for_class = np.mean(target[target==cls], prediction[target==cls])
        class_accuracies.append(acc_for_class)

    return np.mean(class_accuracies)


def mean_absolute_error(prediction, target):
    prediction =  torch_to_np(prediction)
    target = torch_to_np(target)

    # Remove ignore pixels
    prediction = prediction[target != -100]
    target = target[target != -100]

    return np.mean(np.abs(prediction-target))


def _softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=axis,keepdims=True))
    return e_x / np.sum(e_x,axis=axis,keepdims=True)