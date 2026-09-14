"""Offline conversion of retained scientific values; never imported by the running application.

The source workspace is read-only. Migration writes a separate destination so the
original revision history remains available for audit and numerical comparison.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, override

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from scripts.migrate_mechanism_identity import (
    identify_mechanisms,
    remap_references,
    retired_identity_map,
)

type RetiredPayload = dict[str, Any]


def merge_scientific_values(
    latent: RetiredPayload,
    measurement: RetiredPayload | None = None,
    statistical: RetiredPayload | None = None,
) -> ModelSpec:
    """Convert one explicitly resolved historical source context into its canonical value."""
    payload = deepcopy(latent)
    constructs = {item["id"]: item for item in payload["constructs"]}
    edges = {item["id"]: item for item in payload["edges"]}
    indicators = {}
    if measurement is not None:
        content = deepcopy(measurement["measurement_structure"])
        payload["measurement_clock"] = content["model_clock"]
        for indicator in content["indicators"]:
            owner = constructs[indicator.pop("construct_id")]
            owner.setdefault("indicators", []).append(indicator)
            indicators[indicator["id"]] = indicator
        for kind, field in (
            ("known_input", "known_inputs"),
            ("scientific_only", "scientific_only_constructs"),
        ):
            for declaration in deepcopy(measurement[field]):
                owner = constructs[declaration.pop("construct_id")]
                owner["usage"] = {"kind": kind, **declaration}
    if statistical is not None:
        for likelihood in deepcopy(statistical["likelihoods"]):
            indicators[likelihood.pop("indicator_id")]["likelihood"] = likelihood
        for mechanism in deepcopy(statistical["mechanisms"]):
            if "target_id" in mechanism:
                constructs[mechanism.pop("target_id")].setdefault("dynamics", []).append(mechanism)
            else:
                edges[mechanism.pop("edge_id")].setdefault("mechanisms", []).append(mechanism)
        payload["parameters"] = deepcopy(statistical["parameters"])
        for parameter in payload["parameters"]:
            parameter.pop("elements")
            parameter.pop("role")
            parameter.pop("constraint")
            parameter["distribution"] = parameter.pop("prior")
            parameter["distribution_transform"] = parameter.pop("prior_transform")
            parameter.pop("prior_reasoning")
            parameter.pop("prior_sources")
            if parameter["distribution_transform"] in {
                "site_wide",
                "site_row",
                "positive_identity",
            }:
                parameter["distribution_transform"] = "identity"
        payload["policies"] = {
            "initialization": statistical["initialization_policy"],
            "observation_intercept": statistical["observation_intercept_policy"],
        }
    converted, _ = identify_mechanisms(payload)
    from scripts.migrate_component_slots import convert_component_slots

    return ModelSpec.model_validate(convert_component_slots(converted))


def validate_retired_compiler(payload: RetiredPayload, model: ModelSpec) -> None:
    """Check historical priors and anchors before discarding the compiler replica."""
    from nof1_causal_lab.artifacts.execution import AnchorCertificate

    if payload["schema_version"] != 2:
        raise ValueError("This migration consumes compiled schema 2")
    identities = retired_identity_map(model)
    for definition in payload["parameters"]:
        canonical = model.parameter(identities.get(definition["id"], definition["id"]))
        if (
            canonical.distribution is None
            or canonical.model_dump(mode="json")["distribution"] != definition["prior"]
        ):
            raise ValueError(f"Authored and compiled priors disagree for {canonical.id}")
    # Compilation verifies that the complete scientific value can reproduce all
    # active quantities; native coordinates remain local to the historical reader.
    anchors = tuple(
        AnchorCertificate.model_validate(value)
        for value in payload["structure"]["anchor_certificates"]
    )
    if model.check_execution() != anchors:
        raise ValueError("Authored and compiled anchors disagree")


def migrate_fitted(
    payload: bytes, model, retired_compiler, *, array_writer=None, array_loader=None
) -> ModelSpec:
    """Convert historical native draws to a conditioned revision, preserving every joint row."""
    from io import BytesIO
    from pickle import Unpickler

    import jax.numpy as jnp
    import numpy as np

    from nof1_causal_lab.artifacts.identity import ModelRevision
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.inference.persistence import condition_model
    from nof1_causal_lab.models.ssm.inference.types import (
        JointPosteriorDraws,
        ParticleMCMCPosterior,
    )
    from nof1_causal_lab.models.ssm.parameterization import build_site_registry

    class RetiredRecord:
        def __setstate__(self, state):
            self.__dict__.update(state.get("__dict__", state))

    retired_types = {
        ("nof1_causal_lab.models.ssm.model", "SSMSpec"),
        ("nof1_causal_lab.models.ssm.channels", "ObservationSpec"),
        ("nof1_causal_lab.models.ssm.channels", "InputsSpec"),
        ("nof1_causal_lab.artifacts.native_channels", "ObservationChannel"),
        ("nof1_causal_lab.artifacts.native_channels", "InputChannel"),
        ("nof1_causal_lab.models.ssm.inference.types", "FittedArtifact"),
        ("nof1_causal_lab.artifacts.posterior", "PosteriorProvenance"),
    }

    class HistoricalUnpickler(Unpickler):
        @override
        def find_class(self, module, name):
            if (module, name) == ("nof1_causal_lab.artifacts.identity", "CausalDesignRef"):
                return ModelRevision
            if (module, name) in retired_types:
                return RetiredRecord
            return super().find_class(module, name)

    old = HistoricalUnpickler(BytesIO(payload)).load()
    identities = retired_identity_map(model)
    new = {binding.parameter_id: binding for binding in parameter_bindings(model)[0]}
    count = old.result.draws.describe().n_draws
    native = {site.name: np.zeros((count, *site.shape)) for site in build_site_registry(model)}
    covered = set()
    for binding in retired_compiler["parameter_bindings"]:
        pid = identities.get(binding["parameter_id"], binding["parameter_id"])
        for element, coordinate in binding["coordinates"].items():
            identity = identities.get(element, element)
            if identity not in new[pid].coordinates:
                raise ValueError(
                    f"Retired element {element} disagrees with its scientific definition"
                )
            target = new[pid].coordinates[identity]
            native[target.site_name][(slice(None), *target.indices)] = old.result.get_samples()[
                coordinate["site_name"]
            ][(slice(None), *coordinate["indices"])]
            covered.add(identity)
    if covered != {element for binding in new.values() for element in binding.coordinates}:
        raise ValueError("Retired fitted draws do not cover the ModelSpec parameters")
    result = ParticleMCMCPosterior(
        draws=JointPosteriorDraws(
            parameters={name: jnp.asarray(values) for name, values in native.items()},
            latent_paths=old.result.draws.latent_paths,
            state_ids=tuple(retired_compiler["structure"]["spec"]["latent_ids"]),
        ),
        evidence=old.result.evidence,
    )
    return condition_model(
        model, result, times=old.times, array_writer=array_writer, array_loader=array_loader
    )


def migrate_workspace(source: Path, destination: Path) -> RetiredPayload:
    """Copy one offline workspace, resolving every source pin before changing its vocabulary.

    Original files remain untouched. Journal sequences, artifact versions outside
    the merged definition, arrays, and runtime coordinates retain their identity.
    Failed migration leaves no destination. The manifest records each source hash
    and the many-to-one mapping of authored artifacts to ModelSpec revisions.
    """
    import hashlib
    import json
    import shutil
    from tempfile import TemporaryDirectory

    from nof1_causal_lab.artifacts.admission import AdmissionReport
    from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS
    from nof1_causal_lab.artifacts.identification import IdentificationReport
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.machine.artifact_files import artifact_file_spec
    from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo
    from nof1_causal_lab.machine.model_dependencies import MODEL_INPUTS
    from nof1_causal_lab.machine.store import TransitionRecord
    from nof1_causal_lab.models.model_inputs import input_fingerprints

    source, destination = source.resolve(), destination.resolve()
    if not (source / "store").is_dir():
        raise ValueError(f"No versioned store at {source}")
    if destination.exists() or source in destination.parents:
        raise ValueError("Destination must be absent and outside the source workspace")
    if destination.name != source.name:
        raise ValueError("Destination must preserve the workspace ID (directory name)")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def read(path):
        return json.loads(path.read_text())

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    originals = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    metadata = {
        (value["artifact_id"], value["version"]): value
        for path in sorted((source / "store").glob("*/v*/meta.json"))
        for value in [read(path)]
    }
    authored = {
        "latent_structure",
        "measurement_structure",
        "statistical_model_spec",
        "causal_design",
    }
    filenames = {
        "latent_structure": "latent-structure.json",
        "measurement_structure": "measurement_structure.json",
        "statistical_model_spec": "statistical_model_spec.json",
        "causal_design": "causal_design.json",
        "structural_plan": "structural-plan.json",
    }
    contexts: dict[tuple[str, int], tuple[int, int | None, int | None]] = {}

    def payload(aid, version):
        return read(source / "store" / aid / f"v{version}" / filenames[aid])

    def consistent_context(candidates, key):
        if not candidates:
            raise ValueError(f"No explicit scientific source pin for {key}")
        value = max(candidates, key=lambda item: (item[2] is not None, item[1] is not None))
        if any(
            component is not None and component != value[index]
            for candidate in candidates
            for index, component in enumerate(candidate)
        ):
            raise ValueError(f"Inconsistent scientific source pins for {key}")
        return value

    def context(aid, version):
        key = (aid, version)
        if key in contexts:
            return contexts[key]
        pins = metadata[key]["derived_from"]
        if aid == "latent_structure":
            value = (version, None, None)
        elif aid == "measurement_structure":
            value = (pins["latent_structure"], version, None)
        elif aid == "causal_design":
            value = consistent_context(
                [
                    context(parent, pins[parent])
                    for parent in ("latent_structure", "measurement_structure")
                ],
                key,
            )
        else:
            candidates = [
                context(parent, pin)
                for parent, pin in pins.items()
                if parent in authored | {"structural_plan"}
            ]
            value = consistent_context(candidates, key)
            if aid == "statistical_model_spec":
                value = (*value[:2], version)
        contexts[key] = value
        return value

    models: dict[tuple[int, int | None, int | None], tuple[int, ModelSpec, RetiredPayload]] = {}
    mapped: dict[tuple[str, int], tuple[str, int]] = {}
    with TemporaryDirectory(prefix=".additive-migration-", dir=destination.parent) as temp:
        target = Path(temp) / destination.name
        target.mkdir()
        # Only the store and journal are transformed. Traces, source data and scratch
        # records are copied byte-for-byte; old scratch is archival, never resumed.
        for entry in source.iterdir():
            if entry.name in {"store", "episode", "fixture"}:
                continue
            if entry.is_dir():
                shutil.copytree(entry, target / entry.name)
            else:
                shutil.copy2(entry, target / entry.name)
        if (source / "episode" / "traces").exists():
            shutil.copytree(source / "episode" / "traces", target / "episode" / "traces")

        def ensure_model(aid, version):
            key = context(aid, version)
            if key not in models:
                lv, mv, sv = key
                model = merge_scientific_values(
                    payload("latent_structure", lv)["latent_structure"],
                    payload("measurement_structure", mv) if mv is not None else None,
                    payload("statistical_model_spec", sv)["statistical_model_spec"]
                    if sv is not None
                    else None,
                )
                source_aid = (
                    "statistical_model_spec"
                    if sv is not None
                    else "measurement_structure"
                    if mv is not None
                    else "latent_structure"
                )
                source_version = sv if sv is not None else mv if mv is not None else lv
                old = metadata[source_aid, source_version]
                pins = {
                    parent: pin
                    for parent, pin in old["derived_from"].items()
                    if parent not in authored | {"structural_plan", "compiled_ssm"}
                }
                # Resolve original dependencies without recursively treating this
                # ModelSpec's own former stage as a consumer of itself.
                parents = [
                    (parent, pin)
                    for parent, pin in old["derived_from"].items()
                    if parent in authored
                ]
                if parents:
                    base = consistent_context(
                        [context(*pair) for pair in parents], (source_aid, source_version)
                    )
                    parent, pin = next(pair for pair in parents if context(*pair) == base)
                    pins["model"] = ensure_model(parent, pin)
                revision = len(models) + 1
                info = {
                    **old,
                    "artifact_id": "model",
                    "version": revision,
                    "derived_from": pins,
                    "model_inputs": input_fingerprints(model),
                    "consumed_model_inputs": {},
                }
                models[key] = (revision, model, info)
                directory = target / "store" / "model" / f"v{revision}"
                write(directory / "model.json", model.model_dump(mode="json"))
                write(
                    directory / "meta.json",
                    ArtifactVersionInfo.model_validate(info).model_dump(mode="json"),
                )
            return models[key][0]

        for (aid, version), _info in sorted(
            metadata.items(), key=lambda item: (item[1]["created_at"], item[0])
        ):
            if aid in authored:
                mapped[aid, version] = ("model", ensure_model(aid, version))
        by_revision = {revision: model for revision, model, _ in models.values()}
        model_info = {revision: info for revision, _, info in models.values()}

        posterior_models = {}
        inference_logs = {}

        def convert_pins(pins):
            converted = {
                aid: version
                for aid, version in pins.items()
                if aid not in authored | {"structural_plan", "compiled_ssm"}
            }
            scientific = [(aid, version) for aid, version in pins.items() if aid in authored]
            if scientific:
                chosen = consistent_context(
                    [
                        context(aid, version)
                        for aid, version in pins.items()
                        if aid in authored | {"structural_plan"}
                    ],
                    pins,
                )
                owner, version = next(pair for pair in scientific if context(*pair) == chosen)
                converted["model"] = ensure_model(owner, version)
            if "posterior" in converted:
                posterior_version = converted.pop("posterior")
                converted["model"] = posterior_models[posterior_version]
            return converted

        def identification_report(design_version):
            design = payload("causal_design", design_version)["causal_design"]
            report = IdentificationReport(
                outcome=design["latent"]["default_outcome"]["id"]
                if design["latent"]["default_outcome"]
                else None,
                status=design["identifiability"],
            )
            report.validate_model(by_revision[ensure_model("causal_design", design_version)])
            return report.model_dump(mode="json")

        from functools import cache, partial

        from nof1_causal_lab.utils.arrays import read_array, write_array

        array_directory = str(target / "store/arrays")
        conditioned_meta = {}
        for (aid, version), old in sorted(metadata.items()):
            if aid != "posterior":
                continue
            compiler_version = old["derived_from"]["compiled_ssm"]
            compiler_meta = metadata["compiled_ssm", compiler_version]
            base_version = convert_pins(compiler_meta["derived_from"])["model"]
            directory = source / "store/posterior" / f"v{version}"
            value = remap_references(
                read(directory / "diagnostics.json"),
                retired_identity_map(by_revision[base_version]),
            )
            report = InferenceReport.model_validate(
                {key: value[key] for key in InferenceReport.model_fields if key in value}
            )
            pins = {**convert_pins(old["derived_from"]), "model": base_version}
            inference_logs[version] = {
                "input_pins": pins,
                "report": report.model_dump(mode="json"),
                "retained_provenance": value["provenance"],
            }
            if not (directory / "fitted.pkl").exists():
                inference_logs[version]["retention"] = "report_only"
                posterior_models[version] = base_version
                continue
            conditioned = migrate_fitted(
                (directory / "fitted.pkl").read_bytes(),
                by_revision[base_version],
                read(source / "store/compiled_ssm" / f"v{compiler_version}" / "compiled-ssm.json"),
                array_writer=partial(write_array, array_directory),
                array_loader=cache(partial(read_array, array_directory)),
            )
            revision = max(by_revision) + 1
            info = ArtifactVersionInfo.model_validate(
                {
                    "artifact_id": "model",
                    "version": revision,
                    "provenance": old["provenance"],
                    "derived_from": pins,
                    "produced_by": "run:posterior",
                    "created_at": old["created_at"],
                    "model_inputs": input_fingerprints(conditioned),
                }
            ).model_dump(mode="json")
            by_revision[revision], model_info[revision] = conditioned, info
            posterior_models[version] = revision
            mapped[aid, version] = ("model", revision)
            conditioned_meta[aid, version] = info
            write(
                target / "store/model" / f"v{revision}" / "model.json",
                conditioned.model_dump(mode="json"),
            )
            write(target / "store/model" / f"v{revision}" / "meta.json", info)
            inference_logs[version]["engine_evidence"] = {
                "engine": "marginal_particle_gibbs",
                "latent_transition": "euler_maruyama",
            }

        converted_meta = dict(conditioned_meta)
        extraction_outcomes = {}
        for (aid, version), old in metadata.items():
            if aid in authored | {"structural_plan", "posterior"}:
                continue
            pins = convert_pins(old["derived_from"])
            if aid == "compiled_ssm":
                validate_retired_compiler(
                    read(source / "store" / aid / f"v{version}" / "compiled-ssm.json"),
                    by_revision[pins["model"]],
                )
                continue
            if aid == "measurements":
                extraction_outcomes[version] = {
                    "workers": read(source / "store" / aid / f"v{version}" / "measurements.json")[
                        "workers"
                    ],
                    "input_pins": pins,
                    "model_input": model_info[pins["model"]]["model_inputs"]["extraction"],
                }
                continue
            info = {**old, "derived_from": pins, "model_inputs": {}, "consumed_model_inputs": {}}
            if "model" in pins:
                purpose = MODEL_INPUTS[aid]
                info["consumed_model_inputs"] = {
                    purpose: model_info[pins["model"]]["model_inputs"][purpose]
                }
            info = ArtifactVersionInfo.model_validate(info).model_dump(mode="json")
            converted_meta[aid, version] = info
            directory = target / "store" / aid / f"v{version}"
            shutil.copytree(source / "store" / aid / f"v{version}", directory)
            if aid == "identification_report":
                old_design = old["derived_from"]["causal_design"]
                write(directory / "identification_report.json", identification_report(old_design))
            if aid in ARTIFACT_CONTRACTS:
                for filename in artifact_file_spec(aid).json.values():
                    ARTIFACT_CONTRACTS[aid].model_validate(read(directory / filename))
            write(directory / "meta.json", info)

        # Former reports existed only for positive findings. Preserve the negative
        # findings that were embedded in causal designs at their original sequence.
        reported_designs = {
            old["derived_from"]["causal_design"]
            for (aid, _), old in metadata.items()
            if aid == "identification_report"
        }
        next_report = max(
            (version for aid, version in metadata if aid == "identification_report"), default=0
        )
        additional_reports = {}
        for (aid, version), old in metadata.items():
            if aid != "causal_design" or version in reported_designs:
                continue
            next_report += 1
            revision = ensure_model(aid, version)
            info = ArtifactVersionInfo.model_validate(
                {
                    **old,
                    "artifact_id": "identification_report",
                    "version": next_report,
                    "derived_from": {"model": revision},
                    "model_inputs": {},
                    "consumed_model_inputs": {
                        "identification": model_info[revision]["model_inputs"]["identification"]
                    },
                }
            ).model_dump(mode="json")
            additional_reports[version] = info
            directory = target / "store/identification_report" / f"v{next_report}"
            write(directory / "identification_report.json", identification_report(version))
            write(directory / "meta.json", info)

        # Admission findings retain the old statistical artifact version, while its
        # scientific definition maps to the corresponding ModelSpec version.
        for (aid, version), old in metadata.items():
            if aid != "statistical_model_spec":
                continue
            value = payload(aid, version)
            report = AdmissionReport.model_validate(
                {key: value[key] for key in AdmissionReport.model_fields if key in value}
            )
            pins = convert_pins(old["derived_from"])
            pins["model"] = ensure_model(aid, version)
            info = {
                **old,
                "artifact_id": "admission_report",
                "derived_from": pins,
                "model_inputs": {},
                "consumed_model_inputs": {
                    "belief": model_info[pins["model"]]["model_inputs"]["belief"]
                },
            }
            converted_meta["admission_report", version] = info
            write(
                target / "store/admission_report" / f"v{version}" / "admission_report.json",
                report.model_dump(mode="json"),
            )
            write(target / "store/admission_report" / f"v{version}" / "meta.json", info)

        current_model: int = 0
        source_current = {}
        revision_context = {revision: key for key, (revision, _, _) in models.items()}
        for info in conditioned_meta.values():
            revision_context[info["version"]] = revision_context[info["derived_from"]["model"]]
        for path in sorted((source / "episode" / "journal").glob("*.json")):
            record = read(path)
            original_produced = record["produced"]
            original_retracted = record["retracted"]
            move = record["move"]
            if move["kind"] == "run":
                move["operation_id"] = move.pop("artifact_id")
            elif move["kind"] == "write" and move["artifact_id"] in authored:
                move["artifact_id"] = "model"
                move["expected_model_version"] = current_model
            produced: list[RetiredPayload] = []
            seen = set()
            for info in record["produced"]:
                aid, version = info["artifact_id"], info["version"]
                if aid in {"structural_plan", "compiled_ssm"}:
                    continue
                if aid == "posterior":
                    record["diagnostics"].update(inference_logs[version])
                    if (aid, version) not in conditioned_meta:
                        continue
                if aid == "measurements":
                    record["diagnostics"].update(extraction_outcomes[version])
                    continue
                converted = (
                    model_info[ensure_model(aid, version)]
                    if aid in authored
                    else converted_meta[aid, version]
                )
                identity = (converted["artifact_id"], converted["version"])
                if identity not in seen:
                    produced.append(converted)
                    seen.add(identity)
                if aid == "statistical_model_spec":
                    record["diagnostics"]["authored_model"] = payload(aid, version)[
                        "statistical_model_spec"
                    ]
                    produced.append(converted_meta["admission_report", version])
                if aid == "causal_design" and version in additional_reports:
                    produced.append(additional_reports[version])

            retracted = {}
            removed = {
                index: source_current[aid]
                for index, aid in enumerate(
                    ("latent_structure", "measurement_structure", "statistical_model_spec")
                )
                if aid in source_current
                and any(item["artifact_id"] == aid for item in original_retracted)
            }
            affected = current_model and any(
                revision_context[current_model][index] == version
                for index, version in removed.items()
            )
            for item in original_retracted:
                aid = item["artifact_id"]
                if aid in {"posterior", "structural_plan", "compiled_ssm"}:
                    continue
                if aid not in authored:
                    retracted[aid] = item
                if aid == "causal_design":
                    retracted["identification_report"] = {
                        **item,
                        "artifact_id": "identification_report",
                    }
                elif aid == "statistical_model_spec":
                    retracted["admission_report"] = {**item, "artifact_id": "admission_report"}
                if aid in authored - {"causal_design"} and affected:
                    retracted["model"] = {**item, "artifact_id": "model"}
            if affected and not any(info["artifact_id"] == "model" for info in produced):
                # Removing a former detail layer reveals its retained ancestor;
                # it must not delete scientific entities still present in history.
                ancestor = current_model
                while ancestor and any(
                    revision_context[ancestor][index] == version
                    for index, version in removed.items()
                ):
                    ancestor = int(model_info[ancestor]["derived_from"].get("model", 0))
                if ancestor:
                    produced.append(model_info[ancestor])
            record["produced"] = produced
            # Several old artifact names may now describe the same current value.
            # A replacement (including a revealed ancestor) is an installation,
            # not a simultaneous removal of that canonical artifact.
            installed = {info["artifact_id"] for info in produced}
            record["retracted"] = [item for aid, item in retracted.items() if aid not in installed]
            # Accepted scratch from the former submission schema is archived but
            # cannot be resumed into a different authoring contract.
            record["resume"] = None
            record = TransitionRecord.model_validate(record).model_dump(mode="json")
            if record["status"] == "applied":
                for item in original_retracted:
                    source_current.pop(item["artifact_id"], None)
                for info in original_produced:
                    source_current[info["artifact_id"]] = info["version"]
                if any(item["artifact_id"] == "model" for item in record["retracted"]):
                    current_model = 0
                for info in produced:
                    if info["artifact_id"] == "model":
                        current_model = int(info["version"])
            write(target / "episode/journal" / path.name, record)
        manifest = {
            "schema_version": 1,
            "source": str(source),
            "source_sha256": originals,
            "model_revisions": {
                f"{aid}/v{version}": mapped[aid, version][1] for aid, version in sorted(mapped)
            },
            "numerical_outputs": "preserved; no fitting or simulation",
            "restored_identification_reports": {
                str(version): info["version"] for version, info in additional_reports.items()
            },
            "scratch": "archived; previous-schema resumes retired",
        }
        write(target / "migration.json", manifest)
        target.rename(destination)
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = migrate_workspace(args.source, args.destination)
    print(
        f"Migrated {len(manifest['model_revisions'])} authored versions to {args.destination}; originals preserved"
    )


if __name__ == "__main__":
    main()
