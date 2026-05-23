"""Shared constants for the benchmark suite."""

SAMPLE_RATE = 2048
KERNEL_LEN = 1.0
NUM_SAMPLES = int(KERNEL_LEN * SAMPLE_RATE)

IFOS = ["H1", "L1"]
NUM_CHANNELS = len(IFOS)
