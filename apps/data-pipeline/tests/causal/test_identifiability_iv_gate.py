"""Tests for the ``iv_allowed`` gate on ``check_identifiability``.

The flag controls whether graph-theoretic IV candidates can be reported under
an explicit parametric linearity assumption.
"""

from __future__ import annotations

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.utils.identifiability import check_identifiability


def _iv_structure_latent_structure():
    """A DAG with a textbook IV pattern: U → X → Y, Z → X, with U unobserved.

    With U unobserved and confounding both X and Y, the backdoor cannot be
    blocked by adjusting on observed variables. But Z (a parent of X with
    no other path to Y) is a valid instrument *under linearity*.
    """
    return {
        "default_outcome": "construct:Y",
        "constructs": [
            {
                "id": "construct:X",
                "name": "X",
                "temporal_status": "time_invariant",
            },
            {
                "id": "construct:Y",
                "name": "Y",
                "temporal_status": "time_invariant",
            },
            {
                "id": "construct:Z",
                "name": "Z",
                "temporal_status": "time_invariant",
            },
            {
                "id": "construct:U",
                "name": "U",
                "temporal_status": "time_invariant",
            },
        ],
        "edges": [
            {"cause_id": "construct:Z", "effect_id": "construct:X", "lagged": False},
            {"cause_id": "construct:X", "effect_id": "construct:Y", "lagged": False},
            {"cause_id": "construct:U", "effect_id": "construct:X", "lagged": False},
            {"cause_id": "construct:U", "effect_id": "construct:Y", "lagged": False},
        ],
    }


def _measurement_structure_observing_xyz():
    return {
        "indicators": [
            {"name": "y_obs", "construct_id": "construct:Y"},
            {"name": "x_obs", "construct_id": "construct:X"},
            {"name": "z_obs", "construct_id": "construct:Z"},
        ],
    }


class TestIVAllowedDefault:
    def test_default_does_not_promote_a_linear_iv_candidate(self):
        result = check_identifiability(
            _iv_structure_latent_structure(), _measurement_structure_observing_xyz()
        )
        assert "X" not in result["identifiable_treatments"]
        assert "X" in result["non_identifiable_treatments"]
        assert result["graph_info"]["iv_allowed"] is False


class TestIVAllowedFalse:
    def test_disabled_iv_gate_skips_iv(self):
        """With ``iv_allowed=False`` and only-IV-identification structure,
        the treatment must end up non-identifiable."""
        latent_structure = _iv_structure_latent_structure()
        measurement_structure = _measurement_structure_observing_xyz()

        result_with_iv = check_identifiability(
            latent_structure, measurement_structure, iv_allowed=True
        )
        result_no_iv = check_identifiability(
            latent_structure, measurement_structure, iv_allowed=False
        )

        assert result_no_iv["graph_info"]["iv_allowed"] is False

        assert result_with_iv["identifiable_treatments"]["X"]["method"] == "instrumental_variable"
        assert result_with_iv["identifiable_treatments"]["X"]["instruments"] == ["Z"]
        assert "X" not in result_no_iv["identifiable_treatments"]
        assert "X" in result_no_iv["non_identifiable_treatments"]

    def test_disabled_iv_gate_preserves_do_calculus_identifications(self):
        """Treatments identified via do-calculus (backdoor/front-door) should
        be unchanged when IV is disabled — IV is a fallback, not a primary."""
        # Simpler DAG: X → Y, no confounders. Backdoor trivially identifiable.
        latent_structure = {
            "default_outcome": "construct:Y",
            "constructs": [
                {
                    "id": "construct:X",
                    "name": "X",
                    "temporal_status": "time_invariant",
                },
                {
                    "id": "construct:Y",
                    "name": "Y",
                    "temporal_status": "time_invariant",
                },
            ],
            "edges": [{"cause_id": "construct:X", "effect_id": "construct:Y", "lagged": False}],
        }
        measurement_structure = {
            "indicators": [
                {"name": "y_obs", "construct_id": "construct:Y"},
                {"name": "x_obs", "construct_id": "construct:X"},
            ],
        }

        result_with_iv = check_identifiability(
            latent_structure, measurement_structure, iv_allowed=True
        )
        result_no_iv = check_identifiability(
            latent_structure, measurement_structure, iv_allowed=False
        )

        assert "X" in result_with_iv["identifiable_treatments"]
        assert "X" in result_no_iv["identifiable_treatments"]
        # Same method either way.
        assert (
            result_with_iv["identifiable_treatments"]["X"]["method"]
            == result_no_iv["identifiable_treatments"]["X"]["method"]
        )


def test_model_reporting_keeps_nonparametric_findings_without_linear_iv_assumptions():
    from nof1_causal_lab.artifacts.construct import TemporalStatus
    from nof1_causal_lab.models.identification import identify_model
    from tests.helpers import make_model

    model = make_model(["X", "Y", "Z", "U"], [("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")])
    identities = {construct.name: construct.id for construct in model.constructs}
    x_id, y_id, u_id = (identities[name] for name in ("X", "Y", "U"))
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                construct.model_copy(
                    update={
                        "temporal_status": TemporalStatus.TIME_INVARIANT,
                        "indicators": () if construct.id == u_id else construct.indicators,
                    }
                )
                for construct in model.constructs
            ),
        ),
        default_outcome=y_id,
    )
    report = identify_model(model)
    assert x_id not in report.estimable_treatments
    assert x_id in report.non_identifiable

    unconfounded = model.revised(edges=tuple(edge for edge in model.edges if edge.cause.id != u_id))
    identified = identify_model(unconfounded)
    finding = identified.treatments[x_id]
    assert finding.status == "identified"
    assert finding.method == "do_calculus"
