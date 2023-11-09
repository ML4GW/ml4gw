from typing import Literal, Optional

import torch

from ml4gw.gw import snr_frequency_series


class ChiSq(torch.nn.Module):
    def __init__(
        self,
        num_bins: int,
        fftlength: float,
        sample_rate: float,
        highpass: Optional[float] = None,
        return_snr: bool = False,
        input_domain: Literal["time", "frequncy"] = "time",
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate

        # include extra bin so that we have all left and right edges
        self.num_bins = num_bins
        bins = torch.arange(num_bins + 1) / num_bins
        self.register_buffer("bins", bins)

        self.fftsize = int(fftlength * sample_rate)
        self.num_freqs = int(fftlength * sample_rate // 2 + 1)
        freqs = torch.arange(self.num_freqs) / fftlength
        self.register_buffer("freqs", freqs)

        if highpass is not None:
            mask = freqs >= highpass
            self.register_buffer("mask", mask)
        else:
            self.mask = None
        self.return_snr = return_snr
        self.input_domain = input_domain

    def get_cumulative_snr(self, htilde, psd=None, stilde=None):
        """
        Compute the cumulative integral of the SNR frequency
        series along the frequency dimension.
        """

        snr = snr_frequency_series(htilde, self.sample_rate, psd, stilde)
        return snr.cumsum(dim=-1)

    def make_indices(self, batch_size, num_channels):
        """
        Helper function for selecting arbitrary indices
        along the last axis of our batches by building
        tensors of repeated index selectors for the
        batch and channel axes.
        """
        idx0 = torch.arange(batch_size)
        idx0 = idx0.view(-1, 1, 1).repeat(1, num_channels, self.num_bins)

        idx1 = torch.arange(num_channels)
        idx1 = idx1.view(1, -1, 1).repeat(batch_size, 1, self.num_bins)
        return idx0, idx1

    def get_snr_per_bin(self, qtilde, stilde, edges, psd=None):
        """
        For a normalized frequency template qtilde and
        frequency-domain strain measurement stilde, measure
        the SNR in the bins between the specified edges
        (whose last dimension should be one greater than the
        number of bins).
        """

        # calculate how much SNR _actually_ ended up in each bin
        cumulative_snr = self.get_cumulative_snr(qtilde, psd, stilde)

        # since we have the cumulative SNR, all we need to
        # do is grab the value at the left and right bin
        # edges and then subtract them to get the sum
        # in between them
        batch_size, num_channels, _ = cumulative_snr.shape
        idx0, idx1 = self.make_indices(batch_size, num_channels)

        right = cumulative_snr[idx0, idx1, edges[:, :, 1:]]
        left = cumulative_snr[idx0, idx1, edges[:, :, :-1]]

        # need the actual total SNR to see how much
        # we deviated from the expected breakdown
        total_snr = cumulative_snr[:, :, -1:]
        return right - left, total_snr

    def partition_frequencies(self, htilde, psd=None):
        """
        Compute the edges of the frequency bins that would
        (roughly) evenly break up the optimal SNR of the
        template. Normalize the template by its maximum
        SNR as illustrated in TODO: cite
        """
        # compute the cumulative SNR of our template
        # wrt the background PSD as a function of frequency
        cumulative_snr = self.get_cumulative_snr(htilde, psd)

        # break the total SNR up into even bins
        total_snr = cumulative_snr[:, :, -1:]
        bins = self.bins * total_snr

        # figure out which indices along the frequency axis
        # break up the SNR as closely into these bins as possible
        edges = torch.searchsorted(cumulative_snr, bins, side="right")
        edges = edges.clamp(0, cumulative_snr.size(-1) - 1)

        # normalize by the sqrt of the total SNR
        qtilde = htilde / total_snr**0.5
        return qtilde, edges

    def interpolate_psd(self, psd):
        # have to scale the interpolated psd to ensure
        # that the integral of the power remains constant
        factor = (psd.size(-1) / self.num_freqs) ** 2
        psd = torch.nn.functional.interpolate(
            psd, size=self.num_freqs, mode="linear"
        )
        return psd * factor

    def _check_time_domain(self, template, strain):
        bad_tensors, bad_shapes = [], []
        if template.size(-1) != self.fftsize:
            bad_tensors.append("template")
            bad_shapes.append(template.shape)
        if strain.size(-1) != self.fftsize:
            bad_tensors.append("strain")
            bad_shapes.append(strain.shape)

        if bad_tensors:
            verb = "has" if len(bad_tensors) == 1 else "have"
            bad_tensors = " and ".join(bad_tensors)
            raise ValueError(
                "Both template and strain timeseries are "
                "expected to have time dimension of size {}, "
                "but {} {} shape(s) {}".format(
                    self.fftsize, bad_tensors, verb, ",".join(bad_shapes)
                )
            )

    def forward(
        self,
        template: torch.Tensor,
        strain: torch.Tensor,
        psd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Make PSD optional in case strain has already been whitened
        """
        if psd is not None:
            if psd.size(-1) != self.num_freqs:
                psd = self.interpolate_psd(psd)
            if self.mask is not None:
                psd = psd[:, :, self.mask]

        if self.input_domain == "time":
            self._check_time_domain(template, strain)
            htilde = torch.fft.rfft(template, dim=-1) / self.sample_rate
            stilde = torch.fft.rfft(strain, dim=-1) / self.sample_rate
        else:
            htilde, stilde = template, strain
        if self.mask is not None:
            htilde = htilde[:, :, self.mask]
            stilde = stilde[:, :, self.mask]

        qtilde, edges = self.partition_frequencies(htilde, psd)
        snr_per_bin, total_snr = self.get_snr_per_bin(
            qtilde, stilde, edges, psd
        )

        # for each frequency bin, compute the square of the
        # deviation from the expected amount of SNR in the bin
        # and then sum it over all the bins
        chisq_summand = (snr_per_bin - total_snr / self.num_bins) ** 2
        chisq = chisq_summand.sum(dim=-1)

        # normalize by number of degrees of freedom
        chisq *= self.num_bins / (self.num_bins - 1)
        if self.return_snr:
            return chisq, total_snr
        return chisq
