# A collection of DSP utility functions
# Josh Sanz <jsanz@berkeley.edu>
# 2019 02 20

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from matplotlib import pyplot as plt


def golay_80211ad(n):
    """
    Return the a and b golay sequences using the IEEE 802.11ad specification for generation.

    :param n: {128, 64, 32} length of sequence
    :return: (Ga, Gb) tuple
    """
    # Generator polynomials and placeholders
    Wk128 = np.array([-1, -1, -1, -1, +1, -1, -1], dtype=np.int32)
    Dk128 = np.array([ 1,  8,  2,  4,  16, 32, 64], dtype=np.int32)
    Wk64  = np.array([ 1,  1, -1, -1,  1, -1 ], dtype=np.int32)
    Dk64  = np.array([ 2,  1,  4,  8,  16, 32], dtype=np.int32)
    Wk32  = np.array([-1,  1, -1,  1, -1 ], dtype=np.int32)
    Dk32  = np.array([ 1,  4,  8,  2,  16], dtype=np.int32)
    Ga = np.zeros((n,), dtype=np.int8)
    Gb = np.zeros((n,), dtype=np.int8)
    Aprev = np.array([i == 0 for i in range(n)], dtype=np.int32)
    Anext = np.zeros((n,), dtype=np.int32)
    Bprev = np.array([i == 0 for i in range(n)], dtype=np.int32)
    Bnext = np.zeros((n,), dtype=np.int32)
    # Prep based on sequence length
    if n == 128:
        Wk = Wk128
        Dk = Dk128
    elif n == 64:
        Wk = Wk64
        Dk = Dk64
    elif n == 32:
        Wk = Wk32
        Dk = Dk32
    else:
        raise ValueError("n must be in {32, 64, 128}")
    # Generate sequence
    for k in range(len(Wk)):
        Anext[:] = Wk[k] * Aprev + np.concatenate([np.zeros((Dk[k],)), Bprev[:-Dk[k]]])
        Bnext[:] = Wk[k] * Aprev - np.concatenate([np.zeros((Dk[k],)), Bprev[:-Dk[k]]])
        Aprev[:] = Anext[:]
        Bprev[:] = Bnext[:]
    Ga[:] = Anext[-1::-1]
    Gb[:] = Bnext[-1::-1]
    return Ga, Gb


def db20(x):
    """Return 20 log x."""
    return 20 * np.log10(np.abs(x))


def db10(x):
    """Return 10 log x."""
    return 10 * np.log10(np.abs(x))


def nextpow2(x):
    """Return the smallest power of 2 larger than x."""
    return 2**int(np.ceil(np.log2(x)))


def freq_vec(n, Fs=1):
    """Return the normalized fft frequency vector for n samples."""
    if n % 2 == 0:
        f = np.linspace(-0.5 * Fs, (0.5 - 1./n) * Fs, n)
    else:
        f = np.linspace(-0.5 * Fs, 0.5 * Fs, n)
    w = 2 * np.pi * f
    return f, w


def gen_rrc(beta, nfft, samps_per_symb):
    """Return the transfer function of a root raised cosine filter."""
    w = np.linspace(-0.5, 0.5 - 1./nfft, nfft)
    w_max = 0.5 / samps_per_symb
    H = np.zeros((nfft,))
    for i in range(H.size):
        if np.abs(w[i]) < w_max * (1 - beta):
            H[i] = 1
        elif np.abs(w[i]) < w_max * (1 + beta):
            H[i] = 0.5 * (1. + np.cos(np.pi / (2 * w_max * beta) * (np.abs(w[i]) - w_max * (1 - beta))))
    H = ifftshift(np.sqrt(H))
    return H


def gen_rrc_tdomain(beta, nfft, samps_per_symb):
    """Return the impulse response of a root raised cosine filter."""
    return ifft(gen_rrc(beta, nfft, samps_per_symb))


def rrc_interp(x, beta, samps_per_symb):
    """Upsample x with an rrc filter by samps_per_symb."""
    x_upsamp = np.zeros((x.size * samps_per_symb,), dtype=np.complex64)
    x_upsamp[::samps_per_symb] = x
    nfft = 2 ** int(np.ceil(np.log2(x_upsamp.size)))
    Hrrc = gen_rrc(beta, nfft, samps_per_symb)
    # plt.figure()
    f0, _ = freq_vec(n=x.size)
    f, _ = freq_vec(n=nfft)
#     plt.figure()
#     plt.subplot(2,1,1)
#     plt.plot(f, db20(fftshift(fft(x_upsamp, nfft))))
#     plt.subplot(2,1,2)
#     plt.plot(f, np.unwrap(np.angle(fftshift(fft(x_upsamp, nfft)))))
#     plt.plot(f0 / samps_per_symb, np.unwrap(np.angle(fftshift(fft(x)))) * samps_per_symb)
    Y = fft(x_upsamp, nfft) * Hrrc * samps_per_symb
#     plt.figure()
#     plt.subplot(2,1,1)
#     plt.plot(db20(fftshift(Y + 1e-4)))
#     plt.subplot(2,1,2)
#     plt.plot(np.unwrap(np.angle(fftshift(Y))))
    return ifft(Y).astype(np.complex64)


def resample_delayed(x, tau, oversample_rate):
    """Oversample x by oversample_rate, then decimate with a delay."""
    # Ensure we can handle the delay nicely
    assert (x.size * oversample_rate) % 2 == 0, "code assumes oversampled signal has even length"
    assert tau < 1.0, "tau should be a subsample delay"
    assert (round(tau * oversample_rate) == tau * oversample_rate), \
        "tau must be a multiple of 1/oversample_rate"
    delay = int(tau * oversample_rate)
    # FFT upsample
    X = fft(x)
    XX = np.zeros(x.size * oversample_rate, dtype=np.complex64)
    if x.size % 2 == 0:
        XX[0:x.size / 2] = X[0:x.size / 2]
        # Split the extra negative frequency's energy
        XX[x.size / 2 + 1] = X[x.size / 2 + 1] / 2.
        XX[-x.size/2:] = X[x.size / 2:]
        XX[-x.size / 2] *= 0.5
    else:
        n = int(np.floor((x.size - 1) / 2))
        XX[0:n+1] = X[0:n+1]
        XX[-n:] = X[-n]
    XX *= oversample_rate
    xx = ifft(XX)
    return xx[delay::oversample_rate]


def parabolic_fit(Y):
    """Fit a parabola to 3 y values at x=[-1,0,1]; return x0, y0."""
    c = Y[1]
    a = 0.5 * (Y[0] + Y[2]) - Y[1]
    b = 0.5 * (Y[2] - Y[0])
    x0 = -b / (2 * a)
    y0 = - b**2 / (2 * a) + c
    return x0, y0


def fit_freq_domain_delay(x, ref):
    """
    Perform a linear fit to the phase difference.

    A time domain delay corresponds to a linear phase shift with frequency,
    so the slope of the fit gives the time domain delay.
    """
    _, w = freq_vec(x.size)
    phase = np.unwrap(np.angle(fftshift(fft(ref))))
    X = fftshift(fft(x))
    phase_delayed = np.unwrap(np.angle(X))
    weights = np.abs(X)
    phase_diff = phase - phase_delayed
    coeffs = np.polyfit(w, phase_diff, 1, w=weights)
    return -coeffs[0]


def polyphase_core(x, m, f):
    """
    Perform the core algorithm of a polyphase filter.

    x = input data
    m = decimation rate
    f = filter
    """
    # Hack job - append zeros to match decimation rate
    if x.shape[0] % m != 0:
        x = np.append(x, np.zeros((m - x.shape[0] % m,)))
    if f.shape[0] % m != 0:
        f = np.append(f, np.zeros((m - f.shape[0] % m,)))
    # p := polyphase
    p = np.zeros((m, (x.shape[0] + f.shape[0]) / m), dtype=x.dtype)
    p[0, :-1] = np.convolve(x[::m], f[::m])
    # Invert the x values when applying filters
    for i in range(1, m):
        p[i, 1:] = np.convolve(x[m - i::m], f[i::m])
    return p


def polyphase_filter_decimate(x, m, f):
    """Decimate x by m with filter taps f."""
    return np.sum(polyphase_core(x, m, f), axis=0)


def rrc_decimate_pfb(x, beta, samps_per_symb):
    """Decimate x by samps_per_symb with an rrc(beta) filter, using a PFB."""
    hrrc = ifft(gen_rrc(beta, 65, samps_per_symb))
    return polyphase_filter_decimate(x, samps_per_symb, hrrc)


def rrc_decimate_fft(x, beta, samps_per_symb):
    """Decimate x by samps_per_symb with an rrc(beta) filter, using FFTs."""
    nfft = nextpow2(x.size)
    Hrrc = gen_rrc(beta=beta, samps_per_symb=samps_per_symb, nfft=nfft)
    X = fft(x, nfft)
    Y = X * Hrrc
    y = ifft(Y).astype(np.complex64)[::2]
    y = y[:int((x.size + 1) / 2)]
    return y


def cfar_noise_est(data, stencil_width=10, guard_interval=5):
    """CFAR estimation stage with FFT convolution."""
    n_extend = 2 * (stencil_width + guard_interval) + 1
    np2 = int(2**np.ceil(np.log2(data.size + 2 * n_extend)))
    stencil = np.array([1.0] * stencil_width +
                       [0] * (2 * guard_interval + 1) +
                       [1.0] * stencil_width) / (2 * stencil_width)
    stencil_freq = np.conjugate(np.fft.rfft(stencil, np2))
    # Account for edge effects...
    data_extended = np.concatenate([np.flipud(data[0:n_extend]), data,
                                    np.flipud(data[-n_extend:-1])])
    data_freq = np.fft.rfft(np.abs(data_extended), np2)
    return np.fft.irfft(data_freq * stencil_freq)[n_extend//2+1:data.size+n_extend//2+1]
