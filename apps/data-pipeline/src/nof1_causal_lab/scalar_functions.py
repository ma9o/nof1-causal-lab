"""Scientific scalar formulas evaluated with expression operands or numerical arrays."""


def restoring_drift(value, center, stiffness, quartic):
    """The negative gradient of the quadratic and quartic restoring potential."""
    delta = value - center
    return -stiffness * delta - quartic * delta**3


def hill_response(value, emax, ec50, n, *, maximum):
    """Non-negative saturating response with the native denominator stabilization."""
    dose = maximum(value, 0.0)
    powered = dose**n
    return emax * powered / (ec50**n + powered + 1e-12)
