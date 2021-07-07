# # Split the dataset by class values, return a dictionary
def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated

import numpy as np
