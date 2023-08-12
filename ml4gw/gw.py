"""
Tools for manipulating raw gravitational waveforms
and projecting them onto interferometer responses.
Much of the projection code is an extension of the
implementation made available in bilby:

https://arxiv.org/abs/1811.02042

Specifically the code here:
https://github.com/lscsoft/bilby/blob/master/bilby/gw/detector/interferometer.py
"""

from typing import List, Tuple, Union

import torch
from torchtyping import TensorType

from ml4gw.types import (
    NetworkDetectorTensors,
    NetworkVertices,
    PSDTensor,
    ScalarTensor,
    TensorGeometry,
    VectorGeometry,
    WaveformTensor,
)
from ml4gw.utils.interferometer import InterferometerGeometry

SPEED_OF_LIGHT = 299792458.0  # m/s


# define some tensor shapes we'll reuse a bit
# up front. Need to assign these variables so
# that static linters don't give us name errors
batch = num_ifos = polarizations = time = frequency = space = None  # noqa


def outer(x: VectorGeometry, y: VectorGeometry) -> TensorGeometry:
    """
    Compute the outer product of two batches of vectors
    """
    return torch.einsum("...i,...j->...ij", x, y)


def plus(m: VectorGeometry, n: VectorGeometry) -> TensorGeometry:
    return outer(m, m) - outer(n, n)


def cross(m: VectorGeometry, n: VectorGeometry) -> TensorGeometry:
    return outer(m, n) + outer(n, m)


def breathing(m: VectorGeometry, n: VectorGeometry) -> TensorGeometry:
    return outer(m, m) + outer(n, n)


polarization_funcs = {
    "plus": plus,
    "cross": cross,
    "breathing": breathing,
}


def compute_antenna_responses(
    theta: ScalarTensor,
    psi: ScalarTensor,
    phi: ScalarTensor,
    detector_tensors: NetworkDetectorTensors,
    modes: List[str],
) -> TensorType["batch", "polarizations", "num_ifos"]:
    """
    Compute the antenna pattern factors of a batch of
    waveforms as a function of the sky parameters of
    their sources as well as the detector tensors of
    the interferometers whose response is being
    calculated.

    Args:
        theta:
            Angle of each source in radians relative
            to the celestial equator
        psi:
            Angle in radians between each source's
            natural polarization basis and the basis
            which has the 0th unit vector pointing along
            the celestial equator
        phi:
            Angle in radians between each source's right
            ascension and the right ascension of the
            geocenter
        detector_tensors:
            Detector tensor for each of the interferometers
            for which a response is being calculated, stacked
            along the 0th axis
        modes:
            Which polarization modes to compute the response for
    Returns:
        A tensor representing interferometer antenna pattern
            factors for each of the polarizations of each of
            the waveforms, for each interferometer.
    """

    # add a dimension so that we can do some tensor
    # manipulations batch-wise later
    theta = theta.view(-1, 1)
    psi = psi.view(-1, 1)
    phi = phi.view(-1, 1)

    # pre-compute all our trigonometric functions
    # since we'll end up using them all at least twice
    sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
    sin_psi, cos_psi = torch.sin(psi), torch.cos(psi)
    sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)

    # u and v are the unit vectors orthogonal to the
    # source's propogation vector, where v is
    # parallel to the celestial equator
    u = torch.cat(
        [cos_phi * cos_theta, cos_theta * sin_phi, -sin_theta], axis=1
    )
    v = torch.cat([-sin_phi, cos_phi, torch.zeros_like(phi)], axis=1)

    # m and n are the unit vectors along the waveforms
    # natural polarization basis
    m = -u * sin_psi - v * cos_psi
    n = -u * cos_psi + v * sin_psi

    # compute the polarization tensor of each signal
    # as a function of the parameters of its source
    polarizations = []
    for mode in modes:
        try:
            polarization = polarization_funcs[mode](m, n)
        except KeyError:
            raise ValueError(f"No polarization mode {mode}")

        # add a dummy dimension for concatenating
        polarizations.append(polarization)

    # shape: batch x num_polarizations x 3 x 3
    polarization = torch.stack(polarizations, axis=1)

    # compute the weight of each interferometer's response
    # to each polarization: batch x polarizations x ifos
    return torch.einsum("...jk,ijk->...i", polarization, detector_tensors)


def shift_responses(
    responses: WaveformTensor,
    theta: ScalarTensor,
    phi: ScalarTensor,
    vertices: NetworkVertices,
    sample_rate: float,
) -> WaveformTensor:
    omega = torch.column_stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ]
    ).view(-1, 1, 3)

    # compute the time delay between the geocenter
    # and each interferometer in units of s, then
    # convert to units of samples and discretize.
    # Divide by c in the second line so that we only
    # need to multiply the array by a single float
    dt = -(omega * vertices).sum(axis=-1)
    dt *= sample_rate / SPEED_OF_LIGHT
    dt = torch.trunc(dt).type(torch.int64)

    # rolling by gathering implementation based on
    # https://stackoverflow.com/a/68641864
    # start by just creating a big arange along the last axis
    idx = torch.ones_like(responses).type(torch.int64)
    idx = torch.cumsum(idx, axis=-1) - 1

    # apply the offset to the indices along the last axis,
    # then modulo by the waveform size to make all the
    # specified indices legitimate and unique
    idx -= dt[:, :, None]
    idx %= idx.shape[-1]

    # unfortunately I can't figure out how to do this
    # last step without doing looping over the ifos
    rolled = []
    for i in range(len(vertices)):
        ifo = torch.gather(responses[:, i], 1, idx[:, i])
        rolled.append(ifo)
    return torch.stack(rolled, axis=1)


def compute_observed_strain(
    dec: ScalarTensor,
    psi: ScalarTensor,
    phi: ScalarTensor,
    detector_tensors: NetworkDetectorTensors,
    detector_vertices: NetworkVertices,
    sample_rate: float,
    **polarizations: TensorType["batch", "time"],
) -> WaveformTensor:
    """
    Compute the strain timeseries $h(t)$ observed by a network
    of interferometers from the given polarization timeseries
    corresponding to gravitational waveforms from sources with
    the indicated sky parameters.

    Args:
        dec:
            Declination of each source in radians relative
            to the celestial north
        psi:
            Angle in radians between each source's
            natural polarization basis and the basis
            which has the 0th unit vector pointing along
            the celestial equator
        phi:
            Angle in radians between each source's right
            ascension and the right ascension of the
            geocenter
        detector_tensors:
            Detector tensor for each of the interferometers
            for which observed strain is being calculated,
            stacked along the 0th axis
        detector_vertices:
            Vertices for each interferometer's spatial location
            relative to the geocenter. Used to compute delay
            between the waveform observed at the geocenter and
            the one observed at the detector site. To avoid
            adding any delay between the two, reset your coordinates
            such that the desired interferometer is at `(0., 0., 0.)`.
        sample_rate:
            Rate at which the polarization timeseries have been sampled
        polarziations:
            Timeseries for each waveform polarization which
            contributes to the interferometer response. Allowed
            polarizations are `cross`, `plus`, and `breathing`.
    Returns:
        Tensor representing the observed strain at each
        interferometer for each waveform.
    """

    # TODO: just use theta as the input parameter?
    # note that ** syntax is ordered, so we're safe
    # to be lazy and use `list` for the keys and values
    theta = torch.pi / 2 - dec
    antenna_responses = compute_antenna_responses(
        theta, psi, phi, detector_tensors, list(polarizations)
    )

    polarizations = torch.stack(list(polarizations.values()), axis=1)
    waveforms = torch.einsum(
        "...pi,...pt->...it", antenna_responses, polarizations
    )

    return shift_responses(
        waveforms, theta, phi, detector_vertices, sample_rate
    )


def get_ifo_geometry(
    *ifos: str,
) -> Tuple[NetworkDetectorTensors, NetworkVertices]:
    """
    For a given list of interferometer names, retrieve and
    concatenate the associated detector tensors and vertices
    of those interferometers.

    Args:
        ifos: Names of the interferometers whose geometry to retrieve
    Returns:
        A concatenation of the detector tensors of each interferometer
        A concatenation of the vertices of each interferometer
    """

    tensors, vertices = [], []
    for ifo in ifos:
        ifo = InterferometerGeometry(ifo)
        detector_tensor = plus(ifo.x_arm, ifo.y_arm) / 2
        tensors.append(detector_tensor)
        vertices.append(ifo.vertex)

    tensors = torch.stack(tensors)
    vertices = torch.stack(vertices)
    return torch.Tensor(tensors), torch.Tensor(vertices)


def compute_ifo_snr(
    responses: WaveformTensor,
    psd: PSDTensor,
    sample_rate: float,
    highpass: Union[float, TensorType["frequency"], None] = None,
) -> TensorType["batch", "num_ifos"]:
    r"""Compute the SNRs of a batch of interferometer responses

    Compute the signal to noise ratio (SNR) of individual
    interferometer responses to gravitational waveforms with
    respect to a background PSD for each interferometer. The
    SNR of the $i$th waveform at the $j$th interferometer
    is computed as:

    $$\rho_{ij} =
        4 \int_{f_{\text{min}}}^{f_{\text{max}}}
        \frac{\tilde{h_{ij}}(f)\tilde{h_{ij}}^*(f)}
        {S_n^{(j)}(f)}df$$

    Where $f_{\text{min}}$ is a minimum frequency denoted
    by `highpass`, `f_{\text{max}}` is the Nyquist frequency
    dictated by `sample_rate`; `\tilde{h_{ij}}` and `\tilde{h_{ij}}*`
    indicate the fourier transform of the $i$th waveform at
    the $j$th inteferometer and its complex conjugate, respectively;
    and $S_n^{(j)}$ is the backround PSD at the $j$th interferometer.

    Args:
        responses:
            A batch of interferometer responses to a batch of
            raw gravitational waveforms
        psd:
            The one-sided power spectral density of the background
            noise at each interferometer to which a response
            in `responses` has been calculated. If 2D, each row of
            `psd` will be assumed to be the background PSD for each
            channel of _every_ batch element in `responses`. If 3D,
            this should contain a background PSD for each channel
            of each element in `responses`, and therefore the first
            two dimensions of `psd` and `responses` should match.
        sample_rate:
            The frequency at which the waveform responses timeseries
            have been sampled. Upon fourier transforming, should
            match the frequency resolution of the provided PSDs.
        highpass:
            The minimum frequency above which to compute the SNR.
            If a tensor is provided, it will be assumed to be a
            pre-computed mask used to 0-out low frequency components.
            If a float, it will be used to compute such a mask. If
            left as `None`, all frequencies up to `sample_rate / 2`
            will contribute to the SNR calculation.
    Returns:
        Batch of SNRs computed for each interferometer
    """

    # TODO: should we do windowing here?
    # compute frequency power, upsampling precision so that
    # computing absolute value doesn't accidentally zero some
    # values out.
    fft = torch.fft.rfft(responses, axis=-1).type(torch.complex128)
    fft = fft.abs() / sample_rate

    # divide by background asd, then go back to FP32 precision
    # and square now that values are back in a reasonable range
    integrand = fft / (psd**0.5)
    integrand = integrand.type(torch.float32) ** 2

    # mask out low frequency components if a critical
    # frequency or frequency mask was provided
    if highpass is not None:
        if not isinstance(highpass, torch.Tensor):
            freqs = torch.fft.rfftfreq(responses.shape[-1], 1 / sample_rate)
            highpass = freqs >= highpass
        elif len(highpass) != integrand.shape[-1]:
            raise ValueError(
                "Can't apply highpass filter mask with {} frequecy bins"
                "to signal fft with {} frequency bins".format(
                    len(highpass), integrand.shape[-1]
                )
            )
        integrand *= highpass.to(integrand.device)

    # sum over the desired frequency range and multiply
    # by df to turn it into an integration (and get
    # our units to drop out)
    # TODO: we could in principle do this without requiring
    # that the user specify the sample rate by taking the
    # fft as-is (without dividing by sample rate) and then
    # taking the mean here (or taking the sum and dividing
    # by the sum of `highpass` if it's a mask). If we want
    # to allow the user to pass a float for highpass, we'll
    # need the sample rate to compute the mask, but if we
    # replace this with a `mask` argument instead we're in
    # the clear
    df = sample_rate / responses.shape[-1]
    integrated = integrand.sum(axis=-1) * df

    # multiply by 4 for mystical reasons
    integrated = 4 * integrated  # rho-squared
    return torch.sqrt(integrated)


def compute_network_snr(
    responses: WaveformTensor,
    psd: PSDTensor,
    sample_rate: float,
    highpass: Union[float, TensorType["frequency"], None] = None,
) -> ScalarTensor:
    r"""
    Compute the total SNR from a gravitational waveform
    from a network of interferometers. The total SNR for
    the $i$th waveform is computed as

    $$\rho_i = \sqrt{\sum_{j}^{N}\rho_{ij}^2}$$

    where \rho_{ij} is the SNR for the $i$th waveform at
    the $j$th interferometer in the network and $N$ is
    the total number of interferometers.

    Args:
        responses:
            A batch of interferometer responses to a batch of
            raw gravitational waveforms
        backgrounds:
            The one-sided power spectral density of the background
            noise at each interferometer to which a response
            in `responses` has been calculated. If 2D, each row of
            `psd` will be assumed to be the background PSD for each
            channel of _every_ batch element in `responses`. If 3D,
            this should contain a background PSD for each channel
            of each element in `responses`, and therefore the first
            two dimensions of `psd` and `responses` should match.
        sample_rate:
            The frequency at which the waveform responses timeseries
            have been sampled. Upon fourier transforming, should
            match the frequency resolution of the provided PSDs.
        highpass:
            The minimum frequency above which to compute the SNR.
            If a tensor is provided, it will be assumed to be a
            pre-computed mask used to 0-out low frequency components.
            If a float, it will be used to compute such a mask. If
            left as `None`, all frequencies up to `sample_rate / 2`
            will contribute to the SNR calculation.
    Returns:
        Batch of SNRs for each waveform across the interferometer network
    """
    snrs = compute_ifo_snr(responses, psd, sample_rate, highpass)
    snrs = snrs**2
    return snrs.sum(axis=-1) ** 0.5


def reweight_snrs(
    responses: WaveformTensor,
    target_snrs: Union[float, ScalarTensor],
    psd: PSDTensor,
    sample_rate: float,
    highpass: Union[float, TensorType["frequency"], None] = None,
) -> WaveformTensor:
    """Scale interferometer responses such that they have a desired SNR

    Args:
        responses:
            A batch of interferometer responses to a batch of
            raw gravitational waveforms
        target_snrs:
            Either a tensor of desired SNRs for each waveform,
            or a single SNR to which all waveforms should be scaled.
        psd:
            The one-sided power spectral density of the background
            noise at each interferometer to which a response
            in `responses` has been calculated. If 2D, each row of
            `psd` will be assumed to be the background PSD for each
            channel of _every_ batch element in `responses`. If 3D,
            this should contain a background PSD for each channel
            of each element in `responses`, and therefore the first
            two dimensions of `psd` and `responses` should match.
        sample_rate:
            The frequency at which the waveform responses timeseries
            have been sampled. Upon fourier transforming, should
            match the frequency resolution of the provided PSDs.
        highpass:
            The minimum frequency above which to compute the SNR.
            If a tensor is provided, it will be assumed to be a
            pre-computed mask used to 0-out low frequency components.
            If a float, it will be used to compute such a mask. If
            left as `None`, all frequencies up to `sample_rate / 2`
            will contribute to the SNR calculation.
    Returns:
        Rescaled interferometer responses
    """

    snrs = compute_network_snr(responses, psd, sample_rate, highpass)
    weights = target_snrs / snrs
    return responses * weights[:, None, None]
