import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Model edit workbench

    Load a study's current model, edit it as JSON, and submit it as an `edit_model` action.
    The action runs or reuses the specification, identification, data-compatibility, and exact
    predictive checks before it commits the revision. Its findings stay attached to the saved
    model, and any of them can be addressed in a later edit.

    Loading reads the study's local Git snapshot and needs no server. Submitting and polling go
    through the action API, so start the local stack first, as described in the
    [integration-testing guide](../../../docs/guides/agentic_integration_testing.md).
    """)
    return


@app.cell
def imports():
    import json
    import urllib.parse
    import urllib.request

    from nof1_causal_lab.study.snapshots import ModelReader

    return ModelReader, json, urllib


@app.cell(hide_code=True)
def load_md(mo):
    mo.md(r"""
    ## 1. Load a study's model
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


@app.cell(hide_code=True)
def edit_md(mo):
    mo.md(r"""
    ## 2. Edit and submit

    The candidate replaces the whole model. The request carries the loaded revision as
    `expected_revision`, so the action rejects the edit if the study's model changed after it
    was loaded.
    """)
    return


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
        + "/api/studies/"
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


@app.cell(hide_code=True)
def result_md(mo):
    mo.md(r"""
    ## 3. Read the action result

    The action runs asynchronously. Each refresh shows the attempt's messages so far and, once
    it completes, its result body.
    """)
    return


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
