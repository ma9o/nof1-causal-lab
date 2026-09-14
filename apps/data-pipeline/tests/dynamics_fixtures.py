"""Small expression constructors for numerical tests."""

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient, ParameterCoefficient
from nof1_causal_lab.artifacts.expressions import coefficient, hill, restoring_potential, state
from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponentSpec


_FREE = None
_ZERO = FixedCoefficient(value=0)


def _coefficient(slot, role):
    return (
        FixedCoefficient(value=slot.value)
        if isinstance(slot, FixedCoefficient)
        else ParameterCoefficient(parameter_id=scientific_id("parameter", role))
    )


def _state(index):
    return state(f"construct:state{index}")


def _term(expression, target, source=None, *, kind="drift", max_index=None):
    last = max(target, source if source is not None else target, max_index or 0)
    return ExpressionComponentSpec(
        expression=expression,
        target=target,
        source=source,
        state_ids=tuple(f"construct:state{i}" for i in range(last + 1)),
        kind=kind,
    )


def decay_term(target):
    return _term(-coefficient(_coefficient(None, "decay"), "decay") * _state(target), target)


def intercept_term(target, cint=_FREE):
    return _term(coefficient(_coefficient(cint, "intercept"), "intercept"), target)


def potential_term(target, center=_FREE, stiffness=_FREE, quartic=_ZERO):
    return _term(
        restoring_potential(
            _state(target).construct_id,
            center=_coefficient(center, "center"),
            stiffness=_coefficient(stiffness, "decay"),
            quartic=_coefficient(quartic, "quartic"),
        ),
        target,
        kind="potential",
    )


def linear_term(source, target, weight=_FREE):
    return _term(
        coefficient(_coefficient(weight, "weight"), "weight") * _state(source), target, source
    )


def hill_term(source, target, emax=_FREE, ec50=_FREE, n=_FREE):
    return _term(
        hill(
            _state(source),
            emax=_coefficient(emax, "emax"),
            ec50=_coefficient(ec50, "ec50"),
            n=_coefficient(n, "exponent"),
        ),
        target,
        source,
    )


def interaction_term(source_a, source_b, target, weight=_FREE):
    return _term(
        coefficient(_coefficient(weight, "weight"), "weight") * _state(source_a) * _state(source_b),
        target,
        source_a,
        max_index=source_b,
    )
