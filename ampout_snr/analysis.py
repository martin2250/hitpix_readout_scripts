import scipy.optimize
import numpy as np

def fit_peak(wfm: np.ndarray) -> float:
    x_index = np.arange(wfm.size)
    i_max = np.argmax(wfm)
    y_max = wfm[i_max]
    y_cut = y_max * 0.9
    i_right = np.argmax((wfm < y_cut) & (x_index > i_max))
    i_left = np.argmax((wfm[:i_max] < y_cut)*x_index[:i_max])

    y_fit = wfm[i_left:i_right]

    x_fit = np.arange(y_fit.size)
    def parabola(x, a, x0, c):
        return a - c * np.square(x - x0)

    popt, *_ = scipy.optimize.curve_fit(
        f=parabola,
        xdata=x_fit,
        ydata=y_fit,
        p0=(y_max, i_max, 0.01),
    )
    return popt[0]