# Agent-driven model specification

The external agent authors complete or partial [`ModelSpec`](../../pipeline/latent-structure.md#modelspec) values through [`edit_model`](../scientific-actions.md). Read the current model and its immutable revision, edit the relevant scientific definitions, and submit the whole candidate with `expected_revision`. Structure, measurements, mechanisms and laws can change in any order consistent with schema integrity.

## Authoring cycle

1. Read the selected model, data profile and existing findings.
2. Revise the relevant definitions and their referenced parameters or laws together.
3. Submit `edit_model` and poll its attempt until complete.
4. Read typed specification, identification, compatibility and predictive findings from the result body.
5. Decide whether to revise the model, prepare data, fit or request an explicit simulation design.

The backend performs [conditional checks](state-machine.md) without prompting an internal authoring agent. It does not request a construct submission, rationale accepting a soft failure, or repair attempt. Valid incomplete and scientifically problematic candidates remain editable. Invalid schemas and stale expected revisions reject the write.

## Scientific responsibilities

Choose [likelihoods](likelihoods.md) consistent with observed support and recording semantics. Preserve the [identification anchor invariant](identification.md), assign laws on the correct [parameter scales](parameters.md), and use stable scientific identities for references. Graph assumptions include explicit latent confounders. Numeric causal claims still require identification and committed production inference evidence.

Automatic predictive checks use one exact batch of the current whole model when a compatible panel is available. Read its design and law provenance before interpreting findings: fitted-law comparisons on the fitting panel are in-sample posterior predictive checks. Request additional simulation designs or edge-knockout experiments explicitly. Fitting remains an explicit action.

## Logging and history

Action messages contain only timestamp, level and a stable label. Measurements and explanations belong in the typed result body. The model and reports publish together. Subsequent edits reuse reports only when the relevant scientific inputs and check-policy version match; original source revisions remain recorded. Branches and historical reads use the [study history API](../../design/study-history.md#agent-api).
