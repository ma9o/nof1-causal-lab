"""Offline conversion of question artifacts into ordinary ModelSpec revisions."""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from functools import cache, partial
from pathlib import Path
from tempfile import TemporaryDirectory

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo
from nof1_causal_lab.machine.model_dependencies import MODEL_INPUTS
from nof1_causal_lab.machine.store import TransitionRecord
from nof1_causal_lab.models.model_inputs import input_fingerprints
from nof1_causal_lab.utils.arrays import read_array


def fold_questions(workspace: Path) -> dict[str, int]:
    """Convert an offline working copy, retaining journal positions and original input contexts."""

    def read(path):
        return json.loads(path.read_text())

    metadata = {
        (info["artifact_id"], info["version"]): info
        for path in (workspace / "store").glob("*/v*/meta.json")
        for info in [read(path)]
    }
    records = [
        (path, read(path)) for path in sorted((workspace / "episode/journal").glob("*.json"))
    ]
    loader = cache(partial(read_array, str(workspace / "store/arrays")))
    questions = {
        version: read(workspace / "store/question" / f"v{version}/question.json")["text"]
        for aid, version in metadata
        if aid == "question"
    }
    original_models = {
        version: read(workspace / "store/model" / f"v{version}/model.json")
        for aid, version in metadata
        if aid == "model"
    }
    revisions = {}
    contexts = {}
    definitions = {}
    sources = {}
    question_bases = {}
    model_questions = {}
    write_bases = {}

    def add(aid, version, question_version, base):
        key = (aid, version)
        if key in revisions:
            return revisions[key]
        payload = (
            definitions[base].model_dump(mode="json")
            if aid == "question" and base
            else {}
            if aid == "question"
            else deepcopy(original_models[version])
        )
        if question_version is not None:
            payload["question"] = questions[question_version]
        model = ModelSpec.model_validate(payload, context={"distribution_array_loader": loader})
        revision = len(definitions) + 1
        source = deepcopy(metadata[key])
        if aid == "model" and question_version is not None:
            source["derived_from"]["question"] = question_version
        revisions[key], definitions[revision], sources[revision] = revision, model, source
        if aid == "question":
            question_bases[revision] = base
        else:
            model_questions[version] = question_version
            contexts[version, question_version] = revision
        return revision

    current_model = None
    current_question = None
    current_revision = 0
    for path, record in records:
        write_bases[path] = current_revision
        model, question, revision = current_model, current_question, current_revision
        for info in record["produced"]:
            aid, version = info["artifact_id"], info["version"]
            if aid == "question":
                question = version
                revision = add(aid, version, question, revision)
                contexts[model, question] = revision
            elif aid == "model":
                question_pin = info["derived_from"].get("question", question)
                revision = add(aid, version, question_pin, revision)
                model = version
        if record["status"] == "applied":
            current_model, current_question, current_revision = model, question, revision

    # Uncommitted model values remain explicitly version-readable, without being installed.
    def add_uncommitted(version):
        if ("model", version) in revisions:
            return
        pins = metadata["model", version]["derived_from"]
        parent = pins.get("model")
        if parent is not None:
            add_uncommitted(parent)
        question = pins.get("question", model_questions.get(parent))
        add("model", version, question, 0)

    for version in original_models:
        add_uncommitted(version)
    if any(("question", version) not in revisions for version in questions):
        raise ValueError(
            "Question versions without journal context require an explicit model revision"
        )

    def convert_pins(pins):
        result = {aid: version for aid, version in pins.items() if aid not in {"question", "model"}}
        if "question" in pins:
            result["model"] = contexts[pins.get("model"), pins["question"]]
        elif "model" in pins:
            result["model"] = revisions["model", pins["model"]]
        return result

    converted = {}
    for revision, model in definitions.items():
        old = sources[revision]
        pins = (
            ({"model": question_bases[revision]} if question_bases[revision] else {})
            if revision in question_bases
            else convert_pins(old["derived_from"])
        )
        converted["model", revision] = ArtifactVersionInfo.model_validate(
            {
                **old,
                "artifact_id": "model",
                "version": revision,
                "derived_from": pins,
                "model_inputs": input_fingerprints(model),
                "consumed_model_inputs": {},
            }
        ).model_dump(mode="json")
    for (aid, version), old in metadata.items():
        if aid in {"question", "model"}:
            continue
        pins = convert_pins(old["derived_from"])
        consumed = {}
        if "model" in pins and aid in MODEL_INPUTS:
            purpose = MODEL_INPUTS[aid]
            consumed[purpose] = converted["model", pins["model"]]["model_inputs"][purpose]
        converted[aid, version] = ArtifactVersionInfo.model_validate(
            {
                **old,
                "derived_from": pins,
                "consumed_model_inputs": consumed,
            }
        ).model_dump(mode="json")

    rewritten = {}
    for path, original in records:
        record = deepcopy(original)
        move = record["move"]
        if move["kind"] == "write" and move["artifact_id"] in {"question", "model"}:
            move["artifact_id"] = "model"
            move["expected_model_version"] = write_bases[path]
        record["produced"] = [
            converted["model", revisions[info["artifact_id"], info["version"]]]
            if info["artifact_id"] in {"question", "model"}
            else converted[info["artifact_id"], info["version"]]
            for info in record["produced"]
        ]
        diagnostics = record["diagnostics"]
        if "input_pins" in diagnostics:
            diagnostics["input_pins"] = convert_pins(diagnostics["input_pins"])
            if "model_input" in diagnostics:
                version = diagnostics["input_pins"]["model"]
                diagnostics["model_input"] = converted["model", version]["model_inputs"][
                    "extraction"
                ]
        rewritten[path] = TransitionRecord.model_validate(record).model_dump(mode="json")

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    for aid in ("question", "model"):
        directory = workspace / "store" / aid
        if directory.exists():
            shutil.rmtree(directory)
    for (aid, version), info in converted.items():
        directory = workspace / "store" / aid / f"v{version}"
        write(directory / "meta.json", info)
        if aid == "model":
            write(directory / "model.json", definitions[version].model_dump(mode="json"))
    for path, record in rewritten.items():
        write(path, record)
    return {f"{aid}/v{version}": revision for (aid, version), revision in revisions.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if destination.exists() or source in destination.parents:
        parser.error("Destination must be absent and outside the source workspace")
    if source.name != destination.name:
        parser.error("Destination must preserve the workspace ID")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".question-migration-", dir=destination.parent) as temporary:
        target = Path(temporary) / source.name
        shutil.copytree(source, target)
        revisions = fold_questions(target)
        (target / "model-question-migration.json").write_text(
            json.dumps(revisions, indent=2) + "\n"
        )
        target.rename(destination)


if __name__ == "__main__":
    main()
