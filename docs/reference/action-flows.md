# Action flowcharts

What one call to each [scientific action](scientific-actions.md) does, from request to commit. The [Bayesian workflow](../assets/bayesian-workflow.svg) shows how calls follow one another, and the [check catalog](model-checks.md) defines each check.

Besides the failures each chart lists, any call fails without saving when its request is invalid (HTTP `422`, before an attempt starts), when the branch moves before the commit, or on an unexpected error.

## edit_model

![edit_model — save a model revision with the checks its changes require](../assets/action-flows/edit-model.svg)

## prepare_data

![prepare_data — import sources or prepare an observation panel](../assets/action-flows/prepare-data.svg)

## fit

![fit — condition the selected model on an observation panel](../assets/action-flows/fit.svg)

## simulate

![simulate — generate dated histories from the current model](../assets/action-flows/simulate.svg)

## Updating the charts

Each chart is generated from the Mermaid file beside it. Edit the `.mmd` file, then regenerate the Excalidraw scene and SVG with the global `excalidraw-mermaid` tool. Regeneration replaces manual edits to the scenes.

```bash
for action in edit-model prepare-data fit simulate; do
  excalidraw-mermaid --force --require-native --svg "docs/assets/action-flows/$action.svg" \
    "docs/assets/action-flows/$action.mmd" "docs/assets/action-flows/$action.excalidraw"
done
```
