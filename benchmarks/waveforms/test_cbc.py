"""Benchmarks for TaylorF2, IMRPhenomD, and IMRPhenomPv2."""

import torch

from ml4gw.waveforms import IMRPhenomD, IMRPhenomPv2, TaylorF2

F_REF = 20.0
F_MIN = 20.0
F_MAX = 1024.0
DELTA_F = 1.0 / 4.0  # 4-second duration
FREQS = torch.arange(F_MIN, F_MAX, DELTA_F, dtype=torch.float64)


def test_taylorf2_forward(benchmark, cbc_inputs, device, maybe_sync):
    model = TaylorF2().to(device)
    benchmark(maybe_sync(model), FREQS.to(device), **cbc_inputs, f_ref=F_REF)


def test_phenomd_forward(benchmark, cbc_inputs, device, maybe_sync):
    model = IMRPhenomD().to(device)
    benchmark(maybe_sync(model), FREQS.to(device), **cbc_inputs, f_ref=F_REF)


def test_phenompv2_forward(
    benchmark, cbc_inputs, spin_vectors, device, maybe_sync
):
    model = IMRPhenomPv2().to(device)
    (s1x, s1y, s1z), (s2x, s2y, s2z) = spin_vectors
    benchmark(
        maybe_sync(model),
        FREQS.to(device),
        cbc_inputs["chirp_mass"],
        cbc_inputs["mass_ratio"],
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        cbc_inputs["distance"],
        cbc_inputs["phic"],
        cbc_inputs["inclination"],
        F_REF,
    )
