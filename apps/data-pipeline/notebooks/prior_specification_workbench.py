"""Incremental whole-model editing through the scientific action API."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def imports():
    import json
    import urllib.parse
    import urllib.request

    import marimo as mo

    from nof1_causal_lab.machine.snapshots import ModelReader

    return ModelReader, json, mo, urllib


@app.cell(hide_code=True)
def introduction(mo):
    mo.md("""
    # Incremental model workbench

    Load a study, edit its whole model, and submit `edit_model`. The action runs or
    reuses specification, identification, data-compatibility and exact predictive
    checks before committing the revision. Scientific findings remain editable;
    there is no construct admission order or full-model barrier.

    Start the local stack separately using the agentic integration guide. Reading
    a model here uses its local Git snapshot; submitting uses the action API.
    """)
    return


@app.cell
def controls(mo):
    workspace = mo.ui.text(value="DEMO", label="Study")
    backend = mo.ui.text(value="http://localhost:8100", label="Action API")
    load = mo.ui.run_button(label="Load selected model")
    mo.hstack([workspace, backend, load])
    return backend, load, workspace


@app.cell
def selected_model(ModelReader, load, mo, workspace):
    mo.stop(not load.value, mo.md("Load a study to inspect and edit its model."))
    _reader = ModelReader(workspace.value)
    mo.stop(_reader.model is None, mo.md("This study has no model yet."))
    selected_revision = _reader.state.current["model"].revision
    selected_json = _reader.model.model_dump_json(indent=2)
    return selected_json, selected_revision


@app.cell
def candidate_editor(mo, selected_json):
    candidate = mo.ui.text_area(value=selected_json, label="Candidate ModelSpec", rows=24).form(
        submit_button_label="Save model and run applicable checks"
    )
    mo.output.replace(candidate)
    return (candidate,)


@app.cell
def submit_edit(backend, candidate, json, mo, selected_revision, urllib, workspace):
    mo.stop(candidate.value is None)
    _url = (
        backend.value.rstrip("/")
        + "/api/episodes/"
        + urllib.parse.quote(workspace.value, safe="")
        + "/actions"
    )
    _request = urllib.request.Request(
        _url,
        data=json.dumps(
            {
                "action": "edit_model",
                "expected_revision": selected_revision,
                "model": json.loads(candidate.value),
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(_request, timeout=30) as _response:
        receipt = json.load(_response)
    attempt_url = _url + "/" + receipt["attempt_id"]
    mo.md(
        f"Accepted attempt `{receipt['attempt_id']}`. Refresh below to read progress and findings."
    )
    return (attempt_url,)


@app.cell
def poll_control(mo):
    refresh = mo.ui.run_button(label="Refresh action result")
    mo.output.replace(refresh)
    return (refresh,)


@app.cell
def action_result(attempt_url, json, mo, refresh, urllib):
    mo.stop(not refresh.value)
    with urllib.request.urlopen(attempt_url, timeout=30) as _response:
        _result = json.load(_response)
    mo.vstack(
        [
            mo.ui.table(_result["messages"]),
            mo.md("```json\n" + json.dumps(_result["body"], indent=2) + "\n```"),
            mo.md(
                "Completed. Reload the selected model before another edit."
                if _result["done"]
                else "Checks are running; refresh this attempt again."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
