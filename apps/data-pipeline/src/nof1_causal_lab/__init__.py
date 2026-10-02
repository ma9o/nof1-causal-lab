"""Package initialization for the N-of-1 Causal Lab."""

from __future__ import annotations


def _keep_numpyro_argument_checks_opt_in() -> None:
    """Undo NumPyro 0.22's default of validating every constructor's arguments.

    Laws opt in where they are authored or decoded. Internal selectors keep exact
    zero weights as -inf logits, and stored laws record the setting they were built with.
    """
    import numpyro

    numpyro.enable_validation(False)


_keep_numpyro_argument_checks_opt_in()
