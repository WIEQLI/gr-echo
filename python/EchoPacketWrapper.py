#!/usr/bin/env python

# Echo Protocol Packet Header Wrapping and Unwrapping
#
# Josh Sanz <jsanz@berkeley.edu
# 20190408
#

import numpy as np
from matplotlib import pyplot as plt

# from peakdetect import peakdetect
import DSPUtil as dsp


class EchoPacketWrapper:
    """Wraps and unwraps packets for torch_echo."""

    def __init__(self, samps_per_symb, beta_rrc, cfo_samps, cfo_freqs, corr_repetitions=2,
                 guard_samps=64):
        """
        Wrap and unwrap packets in torch_echo gnuradio code.

        Inputs:
        :param samps_per_symb: number of samples per symbol transmitted
        :param rrc_beta: bandwidth expansion factor for root raised cosine filter
        :param cfo_samps: integer number of samples for the CFO correction header portion
        :param cfo_freqs: list of frequencies present in the CFO correction header, Hz
        :param corr_repetitions: number of time the Golay sequence pairs are repeated in the header
        :param guard_samps: number of cyclic prefix samples to insert as a guard interval between
                            packet sections
        """
        self.samps_per_symb = samps_per_symb
        self.beta_rrc = beta_rrc
        self.corr_repetitions = corr_repetitions
        self.corr = self.corr_header(corr_repetitions)
        self.corr_samps = self.corr.size
        self.Ga, self.Gb = dsp.golay_80211ad(128)
        mod = np.exp(0.5j * np.pi * np.arange(128))
        self.Ga_upsamp = dsp.rrc_interp(self.Ga * mod, self.beta_rrc, self.samps_per_symb)
        self.Gb_upsamp = dsp.rrc_interp(self.Gb * mod, self.beta_rrc, self.samps_per_symb)
        self.corr_1sps = np.concatenate([self.Ga * np.exp(0.5j * np.pi * np.arange(128)),
                                         self.Gb * np.exp(0.5j * np.pi * np.arange(128))] *
                                         self.corr_repetitions)
        self.fade_extension = np.ones((30,), dtype=np.complex64)
        self.fade_extension[1::2] *= -1
        # Sigmoid fading
        self.fade_extension *= 1. / (1. + np.exp(np.arange(-15, 15) / 3.))

        self.cfo_samps = cfo_samps
        self.cfo_freqs = cfo_freqs
        self.cfo = self.cfo_header(cfo_samps, cfo_freqs)

        self.guard_samps = guard_samps
        self.corr_offset = 0
        self.body_offset = self.corr_samps + self.guard_samps
        # self.cfo_offset = self.corr_samps + self.guard_samps
        # self.body_offset = self.cfo_offset + self.cfo_samps + self.guard_samps

    def end_corr_offset(self, body_samps):
        """Return the symbol offset of the final correlator field given the body size."""
        return self.body_offset + body_samps + self.guard_samps

    def full_packet_length(self, body_samps):
        """Return the full packet length in symbols given the body size."""
        return self.body_offset + body_samps + self.guard_samps + self.corr_samps

    def body_length(self, full_packet_length):
        """Return the length of the body in symbols given the full packet_length."""
        return full_packet_length - self.body_offset - self.corr_samps - self.guard_samps

    @staticmethod
    def corr_header(repetitions):
        """
        Correlation header generator.

        :param repetitions: number of time the Golay sequence pair is repeated in the field
        :return: header IQ samples
        """
        # assert corr_samps in [64, 128, 256], "corr_samps must be in {64, 128, 256}"
        assert repetitions > 0
        Ga, Gb = dsp.golay_80211ad(128)
        block = np.concatenate([Ga, Gb])
        reps = [block * np.exp(0.5j * np.pi * np.arange(256)) * (-1 ** i) for i in range(repetitions)]
        return np.concatenate(reps).astype(np.complex64)

    @staticmethod
    def cfo_header(cfo_samps, cfo_freqs):
        """
        Carrier frequency offset header generator.

        :param cfo_samps: number of samples in the header
        :param cfo_freqs: frequencies present in the header (should be non-harmonic)
        :return: header IQ samples
        """
        assert len(cfo_freqs) > 0
        cfo = np.zeros((cfo_samps,), dtype=np.complex64)
        for f in cfo_freqs:
            cfo += np.cos(2*np.pi * f * np.arange(cfo_samps))
        cfo /= max(np.abs(cfo))  # Normalize to 1 for transmission by usrp
        return cfo.astype(np.complex64)

    def wrap(self, data):
        """Prepend cfo and channel estimation headers, append second channel estimation header."""
        assert data.size > self.guard_samps
        assembled = np.concatenate([self.corr, data[-self.guard_samps:], data,
                                    self.corr[-self.guard_samps:], self.corr])
        upsamp = dsp.rrc_interp(assembled, self.beta_rrc,
                                self.samps_per_symb)[:assembled.size * self.samps_per_symb]
        return np.concatenate([upsamp, self.fade_extension])

    def unwrap(self, frame, do_plot=False):
        """
        Remove header fields from frame and applies channel & cfo corrections.

        :param do_plot: flag to enable debug plots
        :param frame: aligned samples of echo frame
        :return: cfo-corrected and equalized body of frame
        """
        cfo = self.estimate_cfo([frame[:self.corr_samps], frame[-self.corr_samps:]])
        #cfo = 0
        # print("cfo: {}".format(cfo))
        fsync_frame = EchoPacketWrapper.correct_cfo(frame, cfo)
        corr_field0 = fsync_frame[:self.corr_samps]
        corr_field1 = fsync_frame[-self.corr_samps:]
        hchan, hinv= self.estimate_channel(corr_hdrs=[corr_field0, corr_field1],
                                           do_plot=do_plot)
        # print("hinv: {}".format(hinv))
        # if do_plot:
        #     Hinv = np.fft.fftshift(np.fft.fft(hinv))
        #     plt.plot(np.arange(-0.5, 0.5, 2./self.corr_samps), 20*np.log10(Hinv))
        #     plt.show()
        body_len = self.body_length(frame.size)
        body = EchoPacketWrapper.correct_channel(fsync_frame[self.body_offset:self.body_offset + body_len], hinv)
        if do_plot:
            plt.plot(body.real, body.imag, marker='o', linestyle='none')
            plt.xlabel("Re{body}")
            plt.ylabel("Im{body}")
            plt.title("Constellation of body symbols")
            plt.show()
        return body, hchan  # , cfo, [corr_field0, corr_field1], hinv

    def estimate_channel(self, corr_hdrs, do_plot=False):
        """
        Use golay sequences to estimate channel taps.

        :param corr_hdrs: pre- and post- body channel estimation fields
        :param do_plot: flag to enable debug plots
        :return: inverse of the channel response, smoothed by a hanning window
        """
        mod = np.exp(0.5j * np.pi * np.arange(128))
        sumcorr = np.zeros(128, dtype=np.complex64)
        for this_corr in corr_hdrs:
            for i in range(self.corr_repetitions):
                acorr = np.correlate(-this_corr[256*i:256*i+128], self.Ga * mod, 'same')
                bcorr = np.correlate(-this_corr[256*i+128:256*(i+1)], self.Gb * mod, 'same')
                sumcorr += acorr + bcorr
                if do_plot:
                    plt.plot(np.abs(acorr + bcorr))
                    plt.title("Single Correlation Channel Estimate")
                    plt.show()
        sumcorr /= 2 * self.corr_repetitions * 256
        hann = np.hanning(sumcorr.size)
        sumcorr = np.roll(sumcorr * hann, -64)
        if do_plot:
            plt.plot(np.abs(sumcorr))
            plt.title("Averaged Channel Estimate")
            plt.show()
            plt.plot(np.arange(-0.5, 0.5, 1./sumcorr.size), np.abs(np.fft.fftshift(np.fft.fft(sumcorr))))
            # plt.plot(np.arange(-0.5, 0.5, 1./sumcorr.size), np.abs(np.fft.fftshift(np.fft.fft(hann * sumcorr))))
            plt.title('Averaged Channel Frequency Estimate')
            plt.show()
            plt.plot(np.abs(np.fft.ifft(1 / np.fft.fft(sumcorr) *
                                        np.exp(1j * np.pi * np.arange(128)))))
            plt.title("Inverted Channel")
            plt.show()
        return sumcorr, np.fft.ifft(1 / np.fft.fft(sumcorr) * np.exp(1j * np.pi * np.arange(128)))

    def estimate_cfo(self, corr_hdrs):
        fhats = []
        for corr in corr_hdrs:
            for rep in range(self.corr_repetitions):
                flat_phase = corr[rep * 256:(rep + 1) * 256] * np.conjugate(self.corr_1sps[:256]) # same as dividing since unity gain
                deltas = np.angle(flat_phase[1:] * np.conjugate(flat_phase[:-1]))
                N = flat_phase.size
                k = np.arange(1, N)
                fhat = 1. / (2 * np.pi) * sum(6. * k * (N - k) / (N * (N ** 2 - 1.)) * deltas)
                fhats.append(fhat)
        return np.mean(fhats)

    def estimate_cfo2(self, corr_hdrs):
        fhats = []
        for corr in corr_hdrs:
            # conj mult is same as dividing since unity gain
            flat_phase = corr * np.conjugate(self.corr_1sps)
            angle = np.unwrap(np.angle(flat_phase))
            fhat, _ = np.polyfit(np.arange(angle.size), angle, deg=1, w=np.abs(corr) ** 2)
            fhats.append(fhat / (2 * np.pi))
        return np.mean(fhats)

    @staticmethod
    def correct_channel(body, hinv):
        """
        Apply a channel correction.

        :param body: body of frame to be corrected
        :param hinv: channel inverse
        :return: corrected body
        """
        np2 = int(2 ** np.ceil(np.log2(body.size + hinv.size)))
        tmp = np.fft.ifft(np.fft.fft(hinv, np2) * np.fft.fft(body, np2))[64:64 + body.size]
        return tmp - np.mean(tmp)

    @staticmethod
    def correct_cfo(body, cfo):
        """
        Carrier frequency offset correction.

        :param body: body of frame to be corrected
        :param cfo: normalized carrier frequency offset (1/T)
        :return: frequency-centered body
        """
        return body * np.exp(2j * np.pi * -cfo * np.arange(len(body)))

    def find_channel_estimate_field(self, samps, cfar_threshold, do_plot=False):
        """
        Packet alignment. Finds first correlation peak above the threshold.

        :param samps: a window of samples which should contain a full channel estimation field
        :param threshold: threshold for finding the correlation peak
        :param do_plot: flag to enable debug plots
        :return: index of first sample in the CE field
        """
        # np.save("find_data", samps)
        if samps.size < 256 * self.samps_per_symb:
            return None
        np2 = int(2**np.ceil(np.log2(samps.size - 128 * self.samps_per_symb)))
        acorr = np.fft.ifft(np.fft.fft(samps[:-128 * self.samps_per_symb], np2) *
                            np.conjugate(np.fft.fft(self.Ga_upsamp, np2)))[:(samps.size - 128 *
                                                                             self.samps_per_symb)]
        bcorr = np.fft.ifft(np.fft.fft(samps[128 * self.samps_per_symb:], np2) *
                            np.conjugate(np.fft.fft(self.Gb_upsamp, np2)))[:(samps.size - 128 *
                                                                             self.samps_per_symb)]
        sumcorr = (acorr + bcorr) / 256
        # Since there will be at least two corr headers, use the repetition to accentuate 
        # correlations by multiplying with a delayed copy
        sumcorr = (sumcorr[:-128 * self.samps_per_symb] * 
                   np.conjugate(sumcorr[128 * self.samps_per_symb:]))
        # Add a lower bound on the cfar threshold, otherwise we can get false positives from
        # numeric issues with super low noise.
        # 2**-32 should be well below the minimum value of a 16-bit ADC with some additional
        # processing, but above the 1e-17 values that are sometimes observed.
        cfar = np.clip(dsp.cfar_noise_est(sumcorr, 10, 5), 2**-32, None)
        cfar = np.abs(sumcorr) / cfar
        delays = [d[0] for d in np.argwhere(cfar > cfar_threshold)]  # flatten the list
        if len(delays) < 1:
            return None  # , delays, sumcorr, cfar
        # Search for the true peak on the first detection
        peak = delays[0]
        peakval = cfar[peak]
        i = 1
        # Allow hill-climbing
        while i < len(delays) and (delays[i] - 1 == delays[i - 1] and
                                    cfar[delays[i]] > peakval):
            # Check whether this is a new maximum cfar value
            if cfar[delays[i]] > peakval:
                peak = delays[i]
                peakval = cfar[peak]
            i += 1
        if do_plot:
            plt.plot(np.abs(sumcorr))
            plt.plot(cfar / cfar_threshold)
            if len(delays) > 0:
                plt.scatter(delays, np.abs(cfar[delays] / cfar_threshold), marker='x', color='r')
                plt.scatter(peak, peakval / 10, marker='o', color='k')
            plt.show()
        return peak + 256  # , delays, sumcorr, cfar


class ChannelModel:
    def __init__(self, h, cfo, N0):
        self.h = h
        self.cfo = cfo
        self.N0 = N0

    def apply(self, tx):
        tx0 = tx * np.exp(2j * np.pi * self.cfo * np.arange(len(tx)))
        tx1 = np.convolve(tx0, self.h, 'full')
        n = self.N0 * np.random.randn(len(tx1))
        return tx1 + n


def plot_constellation(iq):
    re = np.real(iq)
    im = np.imag(iq)
    plt.plot(re, im, marker='.', linestyle='none')
    plt.show()


if __name__ == "__main__":
    wrapper = EchoPacketWrapper(2, 0.13, 512, [.09, .23, .41], 1)
    channel = ChannelModel(np.array([1.1, 0.2, -0.1, 0.02]), 0.012, 0.02)
    body = 1 - 2 * np.random.randint(0, 2, (128,))
    tx_frame = wrapper.wrap(body)
    rx_frame = channel.apply(tx_frame)[:wrapper.cfo_samps + 2*wrapper.corr_samps + body.size]
    rx_body = wrapper.unwrap(rx_frame, do_plot=True)
    plot_constellation(rx_frame[512:512+128])
    plot_constellation(rx_body)
    print("bit diffs: {}".format(rx_body - body))
