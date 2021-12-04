from typing import Any, cast

import numpy as np
import scipy.optimize

################################################################################
# sigmoid curve fitting


def fitfunc_sigmoid(x, threshold, noise):
    e = (x - threshold) / noise
    return 1 / (1 + np.exp(-e))

# sigmoid with inverse noise -> speed up fit


def fitfunc_sigmoid_inv(x, threshold, noise):
    e = (x - threshold) * noise
    return 1 / (1 + np.exp(-e))


def fit_sigmoid(injection_voltage: np.ndarray, efficiency: np.ndarray) -> tuple[float, float]:
    # check inputs
    assert len(efficiency.shape) == 1
    if injection_voltage.shape != efficiency.shape:
        print(injection_voltage.shape, efficiency.shape)
    assert injection_voltage.shape == efficiency.shape
    # check for dead / noisy pixels
    if sum(efficiency > 0.3) == 0:  # dead
        return np.inf, np.nan
    if sum(efficiency < 0.7) == 0:  # noisy
        return -np.inf, np.nan
    # get starting values
    i_threshold = np.argmax(
        efficiency > 0.5*(np.min(efficiency) + np.max(efficiency)))
    threshold_0 = injection_voltage[i_threshold]
    # fit
    (threshold, noise), _ = cast(
        tuple[np.ndarray, Any],
        scipy.optimize.curve_fit(
            fitfunc_sigmoid_inv,
            injection_voltage,
            efficiency,
            sigma=2-efficiency*(1-efficiency),
            p0=(threshold_0, 100),
            xtol=1e-3,
            gtol=1e-4,
        ),
    )
    return threshold, 1/noise


def fit_sigmoids(injection_voltage: np.ndarray, efficiency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape_output = efficiency.shape[1:]
    threshold, noise = np.zeros(shape_output), np.zeros(shape_output)
    for idx in np.ndindex(*shape_output):
        efficiency_pixel = efficiency[(..., *idx)]
        try:
            t, w = fit_sigmoid(injection_voltage, efficiency_pixel)
            threshold[idx] = t
            noise[idx] = w
        except RuntimeError:
            threshold[idx] = noise[idx] = np.nan
    return threshold, noise
