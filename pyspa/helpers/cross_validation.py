import numpy as np

def create_kfold_cv_folds(statistics_size, k):
    if k < 2:
        raise ValueError("number of folds must be at least 2")

    fold_indices = np.random.permutation(statistics_size)

    folds = k * [None]
    fold_size = int(np.floor(statistics_size / k))
    for i in range(k - 1):
        folds[i] = np.sort(fold_indices[i * fold_size : (i + 1) * fold_size])
    folds[k - 1] = np.sort(fold_indices[(k - 1) * fold_size:])

    return folds

def get_oos_cv_masks(statistics_size, training_fraction):
    if training_fraction <= 0.0 or training_fraction >= 1.0:
        raise ValueError("training fraction must be between 0 and 1")

    training_size = 1 + int(training_fraction * statistics_size)
    training_mask = np.zeros(statistics_size, dtype="bool")
    training_mask[:training_size] = True
    validation_mask = ~training_mask

    return (training_mask, validation_mask)

def get_kfold_cv_fold_masks(statistics_size, leave_out_indices):
    leave_in_mask = np.ones(statistics_size, dtype="bool")
    leave_in_mask[leave_out_indices] = False
    leave_out_mask = ~leave_in_mask

    return (leave_in_mask, leave_out_mask)
