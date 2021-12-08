from typing import Any, cast

import numpy as np
import scipy.optimize
import scipy.special

################################################################################
# sigmoid curve fitting


def fitfunc_sigmoid(x, threshold, noise):
    e = (x - threshold) / noise
    return scipy.special.expit(e)

# sigmoid with inverse noise -> speed up fit


def fitfunc_sigmoid_inv(x, threshold, noise):
    e = (x - threshold) * noise
    return scipy.special.expit(e)


def fit_sigmoid(injection_voltage: np.ndarray, efficiency: np.ndarray) -> tuple[float, float]:
    # check inputs
    assert efficiency.ndim == 1
    assert injection_voltage.shape == efficiency.shape
    pixel_10p = int(0.1 * efficiency.size) # 10 percent of pixels
    # check for dead / noisy pixels
    if sum(efficiency > 0.3) <= pixel_10p:  # dead
        return np.inf, np.nan
    if sum(efficiency < 0.7) <= pixel_10p:  # noisy
        return -np.inf, np.nan
    if sum(efficiency > 1.1) > pixel_10p:  # probably oscillating
        return np.nan, np.nan
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
