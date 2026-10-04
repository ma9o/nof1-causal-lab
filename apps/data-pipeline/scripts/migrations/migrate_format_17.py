"""Name archived action inputs and remove public branches in a new format-17 copy.

Chain this after migrate_format_16. Never point the destination at a live study.
Uploaded-source hashes must be retained or supplied as verified historical hashes:
--file-hashes JSON maps format-16 call commit OIDs to {filename: sha256}.
--requests supplies verified original arguments for otherwise unknown calls.
Unknown historical attempts remain unknown and cannot be replayed by the viewer.
No scientific execution or numerical reconstruction takes place.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pygit2
from pydantic import TypeAdapter

from nof1_causal_lab.actions.contracts import ScientificActionRequest
from nof1_causal_lab.study.view_models import DataDiffRequest
from nof1_causal_lab.utils.arrays import read_array
from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject, JsonValue

_REQUEST = TypeAdapter(ScientificActionRequest | DataDiffRequest)


def convert_payload(value: JsonValue) -> JsonValue:
    """Branch fields belong to the retired transport, never to scientific values."""
    if isinstance(value, list):
        return [convert_payload(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    return {
        key: convert_payload(item)
        for key, item in value.items()
        if not (key == "branch" and {"seq", "attempt"} <= value.keys())
    }


def migrate(
    source: Path,
    destination: Path,
    *,
    file_hashes: Mapping[str, Mapping[str, str]] | None = None,
    original_requests: Mapping[str, JsonValue] | None = None,
) -> dict[str, str]:
    """Rebuild retained calls using their original request, outputs and execution parent."""
    repo = pygit2.Repository(str(source / "study/history.git"))
    if repo.config.get_int("nof1.format") != 16:
        raise ValueError("Chain migrate_format_17 after migrate_format_16")
    if {name for name in repo.references if name.startswith("refs/heads/")} != {"refs/heads/main"}:
        raise ValueError("This converter requires the single main history used by DEMO and STEPWISE2")
    requests: dict[str, JsonValue] = {}
    sources: dict[str, JsonObject] = {}
    commits = {
        str(commit.id): commit
        for name in repo.references
        if name.startswith(("refs/attempts/", "refs/actions/", "refs/heads/"))
        for commit in repo.walk(repo.references[name].target)
        if "logs/attempt.json" in commit.tree
    }

    def payload(oid: str, name: str):
        return json.loads(repo[pygit2.Oid(hex=oid)].peel(pygit2.Tree)[name].peel(pygit2.Blob).data)

    for oid, commit in sorted(commits.items(), key=lambda item: json.loads(item[1].tree["logs/attempt.json"].peel(pygit2.Blob).data)["seq"]):
        record = json.loads(commit.tree["logs/attempt.json"].peel(pygit2.Blob).data)
        attempt = record["attempt"]
        action, outcome = attempt["action"], attempt["outcome"]
        arguments = attempt["request"]
        if arguments is None and original_requests is not None:
            arguments = original_requests.get(oid)
        applied = outcome["status"] == "applied"
        produced = outcome["effects"]["produced"] if applied else []
        output = {item["artifact_id"]: item for item in produced}
        parent = commit.parents[0].tree
        panel = str(parent["artifacts/panel"].id) if "artifacts/panel" in parent else None
        if arguments is None and applied:
            match action:
                case "set_question":
                    arguments = {"action": action, "question": payload(output["question"]["revision"], "question.json")}
                case "edit_model":
                    model = output["model"]
                    arguments = {"action": action, "expected_revision": model["derived_from"].get("model"), "model": payload(model["revision"], "model.json")}
                case "prepare_data" if "panel" in output:
                    metadata = payload(output["panel"]["revision"], "metadata.json")
                    arguments = {"action": action, "input": {"source": metadata["source"], "definition": metadata["preparation"]} if metadata["kind"] == "file" else metadata["source"]}
                case "fit":
                    result = outcome["result"]
                    sampler = result["report"]["core"]["sampler_diagnostics"]
                    if sampler is not None:
                        settings = sampler["settings"]
                        arguments = {"action": action, "model_revision": result["model"]["revision"], "panel_revision": result["panel"]["revision"], "settings": {name: settings[name] for name in ("num_samples", "num_warmup", "num_chains", "n_particles", "seed")}}
                case "simulate":
                    report = outcome["result"]["report"]
                    arguments = {"action": action, "model_revision": report["model"]["revision"], **report["design"]}
                case "data_diff":
                    report = outcome["result"]["report"]
                    arguments = {"action": action, **{name: refs[0] if len(refs) == 1 else refs for name in ("left", "right") for refs in (report[name],)}}
        if arguments is not None:
            arguments = dict(arguments)
            if action in {"edit_model", "simulate"}:
                # Those executions read the parent's selected panel before format 17.
                arguments["panel_revision"] = panel
            if action == "prepare_data" and "source" in arguments["input"]:
                preparation = arguments["input"]
                selected_source = preparation["source"]
                retained = selected_source.get("hashes")
                hashes = retained if retained else (file_hashes or {}).get(oid)
                if hashes is None:
                    raise ValueError(f"prepare_data {oid} needs verified original file SHA-256 values in --file-hashes")
                selected_source = {**selected_source, "hashes": dict(hashes)}
                arguments["input"] = {**preparation, "source": selected_source}
                if "panel" in output:
                    revision = output["panel"]["revision"]
                    if revision in sources and sources[revision] != selected_source:
                        raise ValueError("The same prepared panel cannot have different source hashes")
                    sources[revision] = selected_source
            if action == "edit_model":
                from scripts.migrations.migrate_format_18 import current_model_definition

                arguments["model"] = current_model_definition(
                    arguments["model"], lambda key: read_array(str(source / "store/arrays"), key)
                )
            parsed = _REQUEST.validate_python(
                arguments,
                context={
                    "distribution_array_loader": lambda key: read_array(
                        str(source / "store/arrays"), key
                    )
                },
            )
            if parsed.action != action:
                raise ValueError(f"Original request for {oid} names another action")
            arguments = parsed.model_dump(mode="json", round_trip=True)
        else:
            print(f"Unknown historical call retained: seq={record['seq']} commit={oid} action={action}; viewer will not replay it")
        requests[str(commit.tree["logs"].id)] = arguments

    def update_file(tree_oid: str, name: str, value):
        if name == "attempt.json":
            return {**value, "attempt": {**value["attempt"], "request": requests[tree_oid]}}
        if name == "metadata.json" and tree_oid in sources:
            return {**value, "source": sources[tree_oid]}
        return value

    return rewrite_study(
        source,
        destination,
        convert_payload,
        update_file=update_file,
        mapping_name="format-17-revisions.json",
        source_format=16,
        target_format=17,
        preserve_model_meaning=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--file-hashes", type=Path, help="Verified original file SHA-256 values keyed by format-16 call commit OID")
    parser.add_argument("--requests", type=Path, help="Verified original arguments keyed by format-16 call commit OID for attempts whose requests were not retained")
    args = parser.parse_args()
    mapping = migrate(args.source.resolve(), args.destination.resolve(), file_hashes=json.loads(args.file_hashes.read_text()) if args.file_hashes is not None else None, original_requests=json.loads(args.requests.read_text()) if args.requests is not None else None)
    print(f"Converted format 16 → 17: {len(mapping)} Git objects into {args.destination}")


if __name__ == "__main__":
    main()
