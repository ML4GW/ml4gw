"""
Various constants, all in SI units.
"""

EulerGamma = 0.577215664901532860606512090082402431

# solar mass
MSUN = 1.988409902147041637325262574352366540e30  # kg

# Geometrized nominal solar mass, m
MRSUN = 1.476625038050124729627979840144936351e3

# Newton's gravitational constant
G = 6.67430e-11  # m^3 / kg / s^2

# Speed of light
C = 299792458.0  # m / s

# pi and 2pi
PI = 3.141592653589793238462643383279502884
TWO_PI = 6.283185307179586476925286766559005768

# G MSUN / C^3 in seconds
gt = G * MSUN / (C**3.0)

# 1 solar mass in seconds. Same value as lal.MTSUN_SI
MTSUN_SI = 4.925490947641266978197229498498379006e-6

# Meters per Mpc.
m_per_Mpc = 3.085677581491367278913937957796471611e22

# 1 Mpc in seconds.
MPC_SEC = m_per_Mpc / C

# Speed of light in vacuum (:math:``c``), in gigaparsecs per second
clightGpc = C / 3.0856778570831e22
