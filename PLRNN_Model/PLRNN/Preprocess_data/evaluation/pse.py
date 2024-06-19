import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


SMOOTHING_SIGMA = 1


def ensure_length_is_even(x):
    n = len(x)
    if n % 2 != 0:
        x = x[:-1]
        n = len(x)
    x = np.reshape(x, (1, n))
    return x

def compute_and_smooth_power_spectrum(x):
    x = ensure_length_is_even(x)
    fft_real = np.fft.rfft(x)
    ps = np.abs(fft_real)**2 * 2 / len(x)
    ps_smoothed = gaussian_filter1d(ps, SMOOTHING_SIGMA)
    return ps_smoothed


def get_power_spectrum(signal):
    # standardize trajectory
    signal = (signal - signal.mean()) / signal.std()
    ps = compute_and_smooth_power_spectrum(signal)
    # normalize to p.m.f.
    return ps / ps.sum()


def power_spectrum_error_per_dim(x_gen, x_true):
    dim_x = x_gen.shape[1]
    pse_per_dim = []
    for dim in range(dim_x):
        ps_true = get_power_spectrum(x_true[:, dim])
        ps_gen = get_power_spectrum(x_gen[:, dim])
        hd = hellinger_distance(ps_true, ps_gen)
        pse_per_dim.append(hd)
    return pse_per_dim


def hellinger_distance(p, q):
    return np.sqrt(1-np.sum(np.sqrt(p*q)))


def power_spectrum_error(x_gen, x_true):
    assert np.all(x_true.shape == x_gen.shape)
    pse_errors_per_dim = power_spectrum_error_per_dim(x_gen, x_true)
    return np.mean(pse_errors_per_dim), pse_errors_per_dim


def plot_spectrum_comparison(spectrum_true, spectrum_gen, plot_save_dir):
    plt.plot(spectrum_true, label='ground truth')
    plt.plot(spectrum_gen, label='generated')
    plt.legend()
    plt.savefig(plot_save_dir)
