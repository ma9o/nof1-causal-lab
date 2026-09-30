"""Move model-owned extraction instructions into data metadata in an offline copy.

Usage: uv run python -m scripts.migrations.migrate_data_preparation SOURCE DESTINATION --files diary.csv
The source workspace remains unchanged. The destination retains branches, attempts,
arrays and original numerical findings; this performs no inference or simulation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pygit2

from nof1_causal_lab.artifacts.data_preparation import (
    DataPreparationSpec,
    DataVariableSpec,
    FileSourceRef,
    PreparedDataMetadata,
    SimulationReplicateRef,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.observations import ObservationSpec
from nof1_causal_lab.machine.git_objects import write_tree
from nof1_causal_lab.models.model_inputs import input_fingerprints
from nof1_causal_lab.utils.arrays import read_array, write_array
from scripts.migrations.migrate_fill_null import update_observation_definitions
from scripts.migrations.study_rewrite import remove_edge_timing

EXTRACTION_FIELDS = {
    "how_to_measure",
    "source_columns",
    "computed_rule",
    "extraction_mode",
    "recording",
    "fill_null",
    "fill_null_limit",
}


def migrate_workspace(
    source: Path,
    destination: Path,
    *,
    files: tuple[str, ...],
    preparation_overrides: dict[str, DataPreparationSpec] | None = None,
) -> dict[str, str]:
    if destination.exists() or destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Choose a new destination outside the source workspace")
    original = pygit2.Repository(str(source / "episode/history.git"))
    if original.config.get_int("nof1.format") != 3:
        raise ValueError("Expected a format-3 study; older formats have no supported migration")
    file_source = FileSourceRef(files=files)
    preparation_overrides = preparation_overrides or {}
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("cache", "scratch"))
    repo = pygit2.Repository(str(destination / "episode/history.git"))
    array_root = str(destination / "store/arrays")
    refs = {name: str(repo.references[name].target) for name in repo.references}
    artifacts = {oid for name, oid in refs.items() if name.startswith("refs/artifacts/")}
    commits = set()
    for name, oid in refs.items():
        if name.startswith(("refs/heads/", "refs/attempts/")):
            for commit in repo.walk(pygit2.Oid(hex=oid)):
                commits.add(str(commit.id))
                if "artifacts" in commit.tree:
                    artifacts.update(
                        str(entry.id) for entry in commit.tree["artifacts"].peel(pygit2.Tree)
                    )
    simulation_sources = {}
    for oid in commits:
        tree = repo[pygit2.Oid(hex=oid)].peel(pygit2.Commit).tree
        if "logs/transition.json" in tree:
            record = json.loads(tree["logs/transition.json"].peel(pygit2.Blob).data)
            if (
                record.get("operation_id") == "simulated_measurements"
                and record["status"] == "applied"
            ):
                for artifact in record["produced"]:
                    if artifact["artifact_id"] == "panel":
                        simulation_sources[artifact["revision"]] = record["diagnostics"][
                            "simulation_source"
                        ]
    profiles = {}
    mapping = {}
    active = set()

    def payload(oid, name):
        return json.loads(repo[pygit2.Oid(hex=oid)].peel(pygit2.Tree)[name].peel(pygit2.Blob).data)

    def tree_files(tree, prefix=""):
        result = {}
        for entry in tree:
            obj = repo[entry.id]
            path = prefix + entry.name
            if isinstance(obj, pygit2.Tree):
                result.update(tree_files(obj, path + "/"))
            else:
                result[path] = obj.peel(pygit2.Blob).data
        return result

    def strip_instructions(value):
        if isinstance(value, dict):
            excluded = (
                EXTRACTION_FIELDS
                if "construct_polarity" in value and "measurement_dtype" in value
                else set()
            )
            return {
                key: strip_instructions(item) for key, item in value.items() if key not in excluded
            }
        if isinstance(value, list):
            return [strip_instructions(item) for item in value]
        return value

    def model_value(oid):
        return ModelSpec.model_validate(
            remove_edge_timing(strip_instructions(payload(oid, "model.json"))),
            context={"distribution_array_loader": lambda ref: read_array(array_root, ref)},
        )

    def preparation(oid):
        original_model = payload(oid, "model.json")
        indicators = {}

        def collect(value):
            if isinstance(value, dict):
                if "construct_polarity" in value and "measurement_dtype" in value:
                    indicators[value["id"]] = value
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(original_model)
        return DataPreparationSpec(
            default_window=original_model["measurement_clock"],
            context=original_model.get("question") or "",
            variables=tuple(
                DataVariableSpec.model_validate(
                    {
                        key: value
                        for key, value in update_observation_definitions(item).items()
                        if key in DataVariableSpec.model_fields
                    }
                )
                for item in indicators.values()
            ),
        )

    def rewrite(value):
        if isinstance(value, dict):
            if {
                "model",
                "design",
                "times",
                "draws",
                "indicator_ids",
                "observations",
            } <= value.keys() and "observation_layout" not in value:
                from nof1_causal_lab.models.ssm.observation_support import (
                    simulation_observation_support,
                )

                model = model_value(value["model"]["revision"])
                support = simulation_observation_support(model, np.asarray(value["times"]))
                shape = (value["draws"], len(value["times"]), len(value["indicator_ids"]))
                mask = np.broadcast_to(
                    np.isfinite(support.support_start_times)
                    & np.isfinite(support.support_end_times),
                    shape,
                )
                value = {
                    **value,
                    "observation_layout": {
                        "variables": [
                            ObservationSpec.model_validate(
                                {
                                    **model.indicator(identity).model_dump(
                                        include=set(ObservationSpec.model_fields)
                                    ),
                                    "observation_window": support.observation_windows[index],
                                }
                            ).model_dump(mode="json")
                            for index, identity in enumerate(value["indicator_ids"])
                        ],
                        "support_start_times": write_array(array_root, support.support_start_times),
                        "support_end_times": write_array(array_root, support.support_end_times),
                        "mask": write_array(array_root, mask),
                    },
                }
            if "artifact_id" in value and "revision" in value and value["revision"] in artifacts:
                revision = migrate(value["revision"])
                return {**payload(revision, "meta.json"), "revision": revision}
            result = {rewrite(key): rewrite(item) for key, item in value.items()}
            if "input_keys" in result and "specification" in result:
                result["input_keys"].pop("data_profile", None)
                result["reused"] = [
                    item for item in result.get("reused", []) if item != "data_profile"
                ]
            return result
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str) and value in artifacts | commits:
            return migrate(value)
        return value

    def convert_artifact(oid, content):
        meta = json.loads(content["meta.json"])
        identity = meta["artifact_id"]
        if identity == "model":
            model = model_value(oid)
            content["model.json"] = model.model_dump_json().encode()
            meta["model_inputs"] = input_fingerprints(model)
        elif identity == "panel":
            model_revision = meta["derived_from"]["model"]
            if meta["produced_by"] == "run:simulated_measurements":
                generating = model_value(model_revision)
                metadata = PreparedDataMetadata(
                    time_origin=None,
                    source=SimulationReplicateRef.model_validate(simulation_sources[oid]),
                    variables=tuple(
                        ObservationSpec.model_validate(
                            {
                                **indicator.model_dump(include=set(ObservationSpec.model_fields)),
                                "observation_window": indicator.observation_window
                                or generating.measurement_clock,
                            }
                        )
                        for indicator in generating.indicators
                    ),
                )
            else:
                recipe = (
                    preparation_overrides[oid]
                    if oid in preparation_overrides
                    else preparation(model_revision)
                )
                metadata = PreparedDataMetadata(
                    time_origin=None,
                    source=file_source,
                    variables=recipe.observation_schema(),
                    preparation=recipe,
                )
            content["metadata.json"] = metadata.model_dump_json(exclude={"time_origin"}).encode()
            meta["derived_from"].pop("model")
            meta["consumed_model_inputs"] = {}
        if identity not in {"model", "panel"} and "model" in meta["derived_from"]:
            from nof1_causal_lab.machine.model_dependencies import MODEL_INPUTS

            purpose = MODEL_INPUTS[identity]
            meta["consumed_model_inputs"] = {
                purpose: input_fingerprints(model_value(meta["derived_from"]["model"]))[purpose]
            }
        content["meta.json"] = json.dumps(meta).encode()
        return content

    def migrate(oid):
        if oid in mapping:
            return mapping[oid]
        if oid in active:
            raise ValueError(f"Cyclic stored reference: {oid}")
        active.add(oid)
        obj = repo[pygit2.Oid(hex=oid)]
        if isinstance(obj, pygit2.Commit):
            parents = [pygit2.Oid(hex=migrate(str(parent.id))) for parent in obj.parents]
            content = {
                key: value
                for key, value in tree_files(obj.tree).items()
                if not key.startswith("artifacts/")
            }
            for name, raw in list(content.items()):
                if name.endswith(".json"):
                    content[name] = json.dumps(rewrite(json.loads(raw)), sort_keys=True).encode()
            tree = repo.TreeBuilder(write_tree(repo, content))
            if "artifacts" in obj.tree:
                builder = repo.TreeBuilder()
                for entry in obj.tree["artifacts"].peel(pygit2.Tree):
                    builder.insert(
                        entry.name, pygit2.Oid(hex=migrate(str(entry.id))), pygit2.GIT_FILEMODE_TREE
                    )
                if "panel" in obj.tree["artifacts"].peel(pygit2.Tree):
                    panel = migrate(str(obj.tree["artifacts/panel"].id))
                    builder.insert(
                        "data_profile", pygit2.Oid(hex=profiles[panel]), pygit2.GIT_FILEMODE_TREE
                    )
                tree.insert("artifacts", builder.write(), pygit2.GIT_FILEMODE_TREE)
            result = repo.create_commit(
                None, obj.author, obj.committer, obj.message, tree.write(), parents
            )
        else:
            content = convert_artifact(oid, tree_files(obj.peel(pygit2.Tree)))
            for name, raw in list(content.items()):
                if name.endswith(".json"):
                    content[name] = json.dumps(rewrite(json.loads(raw)), sort_keys=True).encode()
            result = write_tree(repo, content)
            if json.loads(content["meta.json"])["artifact_id"] == "panel":
                import polars as pl

                from nof1_causal_lab.flows.transitions.validation.flow import profile_data
                from nof1_causal_lab.machine.store import utc_now_iso

                table = (
                    destination
                    / "store/blobs"
                    / json.loads(content["external.json"])["panel.parquet"]
                )
                report = profile_data(
                    pl.read_parquet(table),
                    metadata=PreparedDataMetadata.model_validate(
                        {**json.loads(content["metadata.json"]), "time_origin": None}
                    ),
                )
                profile_meta = {
                    "artifact_id": "data_profile",
                    "derived_from": {"panel": str(result)},
                    "produced_by": "migration:data_profile",
                    "created_at": utc_now_iso(),
                    "model_inputs": {},
                    "consumed_model_inputs": {},
                }
                profile = write_tree(
                    repo,
                    {
                        "data_profile.json": report.model_dump_json().encode(),
                        "meta.json": json.dumps(profile_meta).encode(),
                    },
                )
                profiles[str(result)] = str(profile)
                repo.references.create(
                    f"refs/artifacts/data_profile/{profile}", profile, force=True
                )
        mapping[oid] = str(result)
        active.remove(oid)
        return str(result)

    for name, oid in refs.items():
        result = migrate(oid)
        if name.startswith("refs/artifacts/"):
            repo.references.delete(name)
            name = name.rsplit("/", 1)[0] + "/" + result
        repo.references.create(name, pygit2.Oid(hex=result), force=True)
    repo.config["nof1.format"] = 4
    (destination / "data-preparation-mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument(
        "--preparations-json",
        type=Path,
        help="Explicit panel-revision to preparation-spec mapping for archived panels with a broader schema than their pinned model",
    )
    args = parser.parse_args()
    overrides = (
        {
            key: DataPreparationSpec.model_validate(value)
            for key, value in json.loads(args.preparations_json.read_text()).items()
        }
        if args.preparations_json
        else {}
    )
    result = migrate_workspace(
        args.source, args.destination, files=tuple(args.files), preparation_overrides=overrides
    )
    print(f"Migrated {len(result)} objects to {args.destination}; source unchanged")


if __name__ == "__main__":
    main()
