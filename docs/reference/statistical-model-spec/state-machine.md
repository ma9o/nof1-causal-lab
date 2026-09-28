# Incremental check execution

Model edits no longer pass through a construct-admission state machine. The durable [episode workflow](../../../apps/data-pipeline/src/nof1_causal_lab/machine/temporal/workflow.py) serializes actions, checks revisions, runs their activities and atomically publishes their results. The scientific control flow belongs to the external agent.

See the four [action flowcharts](../action-flows.md) for the complete request-to-result paths, including check reuse and automatic simulation.

## Fixed check sequence

1. Stage the submitted whole model without advancing the branch.
2. Check execution completeness and fitting-law capability.
3. Refresh causal identification when its graph or measurement inputs changed.
4. Refresh the selected panel profile when the panel changed.
5. Check model/data compatibility when either relevant input changed.
6. Run one shared exact predictive batch when model and compatible panel prerequisites hold and its input key changed.
7. Publish the model, findings and timestamped action labels in one commit.

[Check selection](../../../apps/data-pipeline/src/nof1_causal_lab/actions/model_checks.py) is a fixed sequence, with no topology scheduler, construct frontier, acceptance state, repair loop or checkpoint rebase. Each group records a fingerprint of its scientific inputs and check-policy version. Reuse is local to the selected study snapshot and preserves the source revisions of the original report. A policy change invalidates reuse even when scientific inputs are unchanged.

## Predictive checks

The [shared generator](../../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/simulation.py) samples current scalar or joint laws, preserving joint dependence. It uses true nonlinear Diffrax dynamics and the declared emission density. Automatic checks use the compatible panel's schedule, 200 draws and seed 0. Edge-knockout contrasts remain an explicit `simulate` choice because each edge adds another batch.

Missing prerequisites are `not_evaluated` findings with typed reasons. Finiteness failures are saved scientific findings; reductions that depend on finite paths are skipped. Unexpected compiler or worker failures fail the action. A fitting-law limitation does not suppress simulation when the current law supports it.

`prepare_data` runs the same affected model/panel checks after publishing its staged panel to the evaluator. `fit` refreshes inexpensive findings without launching a predictive batch. Reports and [action labels](../scientific-actions.md#dispatch-and-polling) remain readable from committed study history.

## Durability

Temporal retains retry, polling and crash recovery for each action. A dedicated model-check worker limits heavy checks to one concurrent activity per worker. Git owns branch conflict checks and publication. A failed action retains its attempted outcome without advancing the scientific branch; a negative scientific finding accompanies a saved model.
