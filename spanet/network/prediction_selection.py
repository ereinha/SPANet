from typing import List
import numpy as np

import numba
from numba import njit

TArray = np.ndarray

TFloat32 = numba.types.float32
TInt64 = numba.types.int64

TPrediction = numba.typed.typedlist.ListType(TFloat32[::1])
TPredictions = numba.typed.typedlist.ListType(TFloat32[:, ::1])

TIResult = TInt64[:, ::1]
TIResults = TInt64[:, :, ::1]

TFResult = TFloat32[:, ::1]
TFResults = TFloat32[:, :, ::1]

NUMBA_DEBUG = False


if NUMBA_DEBUG:
    def njit(*args, **kwargs):
        def wrapper(function):
            return function
        return wrapper


@njit("void(float32[::1], int64, int64, float32)")
def mask_1(data, size, index, value):
    data[index] = value


@njit("void(float32[::1], int64, int64, float32)")
def mask_2(flat_data, size, index, value):
    data = flat_data.reshape((size, size))
    data[index, :] = value
    data[:, index] = value


@njit("void(float32[::1], int64, int64, float32)")
def mask_3(flat_data, size, index, value):
    data = flat_data.reshape((size, size, size))
    data[index, :, :] = value
    data[:, index, :] = value
    data[:, :, index] = value


# @njit("void(float32[::1], int64, int64, float32)")
# def mask_4(flat_data, size, index, value):
#     data = flat_data.reshape((size, size, size, size))
#     data[index, :, :, :] = value
#     data[:, index, :, :] = value
#     data[:, :, index, :] = value
#     data[:, :, :, index] = value


# @njit("void(float32[::1], int64, int64, float32)")
# def mask_5(flat_data, size, index, value):
#     data = flat_data.reshape((size, size, size, size, size))
#     data[index, :, :, :, :] = value
#     data[:, index, :, :, :] = value
#     data[:, :, index, :, :] = value
#     data[:, :, :, index, :] = value
#     data[:, :, :, :, index] = value
#
#
# @njit("void(float32[::1], int64, int64, float32)")
# def mask_6(flat_data, size, index, value):
#     data = flat_data.reshape((size, size, size, size, size, size))
#     data[index, :, :, :, :, :] = value
#     data[:, index, :, :, :, :] = value
#     data[:, :, index, :, :, :] = value
#     data[:, :, :, index, :, :] = value
#     data[:, :, :, :, index, :] = value
#     data[:, :, :, :, :, index] = value


# @njit("void(float32[::1], int64, int64, float32)")
# def mask_7(flat_data, size, index, value):
#     data = flat_data.reshape((size, size, size, size, size, size, size))
#     data[index, :, :, :, :, :, :] = value
#     data[:, index, :, :, :, :, :] = value
#     data[:, :, index, :, :, :, :] = value
#     data[:, :, :, index, :, :, :] = value
#     data[:, :, :, :, index, :, :] = value
#     data[:, :, :, :, :, index, :] = value
#     data[:, :, :, :, :, :, index] = value
#
#
# @njit("void(float32[::1], int64, int64, float32)")
# def mask_8(flat_data, size, index, value):
#     data = flat_data.reshape((size, size, size, size, size, size, size, size))
#     data[index, :, :, :, :, :, :, :] = value
#     data[:, index, :, :, :, :, :, :] = value
#     data[:, :, index, :, :, :, :, :] = value
#     data[:, :, :, index, :, :, :, :] = value
#     data[:, :, :, :, index, :, :, :] = value
#     data[:, :, :, :, :, index, :, :] = value
#     data[:, :, :, :, :, :, index, :] = value
#     data[:, :, :, :, :, :, :, index] = value


@njit("void(float32[::1], int64, int64, int64, float32)")
def mask_jet(data, num_partons, max_jets, index, value):
    if num_partons == 1:
        mask_1(data, max_jets, index, value)
    elif num_partons == 2:
        mask_2(data, max_jets, index, value)
    elif num_partons == 3:
        mask_3(data, max_jets, index, value)
    # elif num_partons == 4:
    #     mask_4(data, max_jets, index, value)
    # elif num_partons == 5:
    #     mask_5(data, max_jets, index, value)
    # elif num_partons == 6:
    #     mask_6(data, max_jets, index, value)
    # elif num_partons == 7:
    #     mask_7(data, max_jets, index, value)
    # elif num_partons == 8:
    #     mask_8(data, max_jets, index, value)


@njit("int64[::1](int64, int64)")
def compute_strides(num_partons, max_jets):
    strides = np.zeros(num_partons, dtype=np.int64)
    strides[-1] = 1
    for i in range(num_partons - 2, -1, -1):
        strides[i] = strides[i + 1] * max_jets

    return strides


@njit(TInt64[::1](TInt64, TInt64[::1]))
def unravel_index(index, strides):
    num_partons = strides.shape[0]
    result = np.zeros(num_partons, dtype=np.int64)

    remainder = index
    for i in range(num_partons):
        result[i] = remainder // strides[i]
        remainder %= strides[i]
    return result


@njit(TInt64(TInt64[::1], TInt64[::1]))
def ravel_index(index, strides):
    return (index * strides).sum()


@njit(numba.types.Tuple((TInt64, TInt64, TFloat32))(TPrediction))
def maximal_prediction(predictions):
    best_jet = -1
    best_prediction = -1
    best_value = -np.float32(np.inf)

    for i in range(len(predictions)):
        max_jet = np.argmax(predictions[i])
        max_value = predictions[i][max_jet]

        if max_value > best_value:
            best_prediction = i
            best_value = max_value
            best_jet = max_jet

    return best_jet, best_prediction, best_value


@njit(numba.types.Tuple((TIResult, TFResult))(TPrediction, TInt64[::1], TInt64))
def extract_prediction(predictions, num_partons, max_jets):
    float_negative_inf = -np.float32(np.inf)
    max_partons = num_partons.max()
    num_targets = len(predictions)

    # Create copies of predictions for safety and calculate the output shapes
    strides = []
    for i in range(num_targets):
        strides.append(compute_strides(num_partons[i], max_jets))

    # Fill up the prediction matrix
    # -2 : Not yet assigned
    # -1 : Masked value
    # else : The actual index value
    results = np.zeros((num_targets, max_partons), np.int64) - 2
    results_weights = np.zeros((num_targets, max_partons), dtype=np.float32) - np.float32(np.inf)

    for _ in range(num_targets):
        best_jet, best_prediction, best_value = maximal_prediction(predictions)

        if not np.isfinite(best_value):
            return results, results_weights

        best_jets = unravel_index(best_jet, strides[best_prediction])

        results[best_prediction, :] = -1
        results_weights[best_prediction, :] = float_negative_inf
        for i in range(num_partons[best_prediction]):
            results[best_prediction, i] = best_jets[i]
            results_weights[best_prediction, i] = best_value

        predictions[best_prediction][:] = float_negative_inf
        for i in range(num_targets):
            for jet in best_jets:
                mask_jet(predictions[i], num_partons[i], max_jets, jet, float_negative_inf)

    return results, results_weights


@njit(numba.types.Tuple((TIResults, TFResults))(TPredictions, TInt64[::1], TInt64, TInt64), parallel=True)
def _extract_predictions(predictions, num_partons, max_jets, batch_size):
    output = np.zeros((batch_size, len(predictions), num_partons.max()), np.int64)
    weight = np.zeros((batch_size, len(predictions), num_partons.max()), np.float32)
    predictions = [p.copy() for p in predictions]

    for batch in numba.prange(batch_size):
        current_prediction = numba.typed.List([prediction[batch] for prediction in predictions])
        output[batch, :, :], weight[batch, :, :] = extract_prediction(current_prediction, num_partons, max_jets)

    return np.ascontiguousarray(output.transpose((1, 0, 2))), np.ascontiguousarray(weight.transpose((1, 0, 2)))


def extract_predictions(predictions: List[TArray]):
    flat_predictions = numba.typed.List([p.reshape((p.shape[0], -1)) for p in predictions])
    num_partons = np.array([len(p.shape) - 1 for p in predictions])
    max_jets = max(max(p.shape[1:]) for p in predictions)
    batch_size = max(p.shape[0] for p in predictions)

    max_partons = np.max(num_partons)
    results = np.zeros((max_partons, len(num_partons), max_partons))
    weights = np.zeros((max_partons, len(num_partons), max_partons))
    for i in range(num_partons):
        temp_predictions = flat_predictions.copy()
        slice = temp_predictions[:,i*jets*jets:(i+1*jets*jets)]
        for j in range(len(slice)):
            mini_slice = slice[j]
            max_indices = np.argmax(mini_slice, axis=1)
            mini_slice[max_indices] = 999
            temp_predictions[j,i*jets*jets:(i+1*jets*jets)] = slice
        result, weight = _extract_predictions(temp_predictions, num_partons, max_jets, batch_size)
        results[i,:,:] = result
        weights[i,:,:] = weight
        
    max_weight_indices = np.argmax(np.prod(weights, axis=0), axis=0)
    max_results = np.take_along_axis(results, np.expand_dims(max_weight_indices, axis=0), axis=0).squeeze()
        
    return [max_result[:partons] for max_result, partons in zip(max_results.T, num_partons)]
