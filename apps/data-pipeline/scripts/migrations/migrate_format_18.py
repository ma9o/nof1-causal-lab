"""Translate format 17 into scientific facts and production-labelled joint laws.

Chain after migrate_format_17. Only a new destination is written; array and table
blobs are copied unchanged. Stored reports/checks are dropped, data_diff leaves
remain, and report-only fits retain their outcome/messages with no numerical result.
Compilation is used once to recover missing historical coordinates, never to fit,
sample, validate findings, or manufacture native telemetry.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygit2

from nof1_causal_lab.artifacts.identity import ConstructId
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm.compile.bindings import joint_law_layout
from nof1_causal_lab.numpyro_json import distribution_shape
from nof1_causal_lab.utils.arrays import read_array, write_array
from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject, JsonValue
    from nof1_causal_lab.numpyro_json import ArrayLoader

_DERIVED = frozenset({"identification_report", "data_profile", "validation_report"})
_SAMPLER_COPIES = frozenset(
    {
        "latent_kernel",
        "latent_smoother",
        "latent_smoother_algorithm",
        "latent_smoother_family",
        "latent_smoother_selection",
        "latent_smoother_parallel",
        "latent_backward_sampling",
        "amala_delta_adapted",
        "latent_transition_kind",
        "latent_init_method",
    }
)
_PROFILE_REMOVALS = frozenset(
    {
        "measurement_dtype",
        "mean",
        "std",
        "min",
        "max",
        "q25",
        "q50",
        "q75",
        "variance",
        "time_coverage_ratio",
        "max_gap_ratio",
        "dtype_violations",
        "duplicate_pct",
        "arithmetic_sequence_detected",
        "n_unparseable_timestamps",
        "zero_fraction",
        "is_nonnegative",
        "is_unit_interval",
        "looks_integer_valued",
        "variance_to_mean_ratio",
    }
)


def convert_payload(value: JsonValue) -> JsonValue:
    """Owner-specific field removals; homonymous authored fields and wire tags stay."""
    if isinstance(value, list):
        return [convert_payload(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    drop: set[str] = set()
    if {"artifact_id", "derived_from"} <= value.keys():
        drop.update(("model_inputs", "consumed_model_inputs"))
        value = {
            **value,
            "derived_from": {
                key: item for key, item in value["derived_from"].items() if key not in _DERIVED
            },
        }
    if {"produced", "retracted"} <= value.keys():
        drop.add("checks")
        value = {
            **value,
            **{
                key: [item for item in value[key] if item["artifact_id"] not in _DERIVED]
                for key in ("produced", "retracted")
            },
        }
    if {"kind", "source", "time_origin"} <= value.keys() and value["kind"] in {
        "file",
        "simulation",
    }:
        drop.add("variables" if value["kind"] == "file" else "preparation")
    if (
        value.get("status") == "identified"
        and "estimand" in value
        or {"treatment", "outcome", "estimand"} <= value.keys()
    ):
        drop.add("method")
    if {"n_samples", "duration_seconds"} <= value.keys():
        drop.add("method")
    if "assessments" in value and "scope" in value:
        drop.add("scope")
    if {"elpd_loo", "p_loo"} <= value.keys():
        drop.update(("observation_unit", "prediction_task", "likelihood_source"))
    if value.get("engine") == "marginal_particle_gibbs" and "latent_transition" in value:
        drop.update(("engine", "latent_transition"))
    if {"probabilities", "n_draws"} <= value.keys():
        drop.add("kind")
    if value.get("kind") in ("authored", "mixed", "unknown") and "interpretation" in value:
        drop.add("interpretation")
    if {"id", "name", "arguments"} <= value.keys() and value.get("type") == "function":
        drop.add("type")
    if {"trace_data", "rank_histograms"} <= value.keys():
        drop.update(("tempering", "posterior_pairs"))
    if "subject" in value and "chains" in value:
        drop.update(("parameter", "n_bins"))
    if {"settings", "parameter_kernel"} <= value.keys():
        drop.update(_SAMPLER_COPIES)
    if {"n_parameter_particles", "latent_delta"} <= value.keys():
        drop.update(("latent_smoother", "latent_init_method"))
    if {"n_draws", "state_ids"} <= value.keys():
        drop.update(("parameter_shapes", "latent_shape"))
    if {"parameter", "code", "suggested_adjustment"} <= value.keys():
        drop.update(
            (
                "supporting_codes",
                "repair_scope",
                "bad_sample_sites",
                "pathology_certificate",
                "is_valid",
                "origin",
                "severity",
                "bad_manifest_names",
                "failing_draw_indices",
                "first_bad_time_index",
            )
        )
    if {"n_obs", "measurement_dtype", "variance"} <= value.keys():
        drop.update(_PROFILE_REMOVALS)
    if {"value", "probability", "count"} <= value.keys():
        drop.add("count")
    if {"times", "support_start", "support_end", "empirical"} <= value.keys():
        drop.add("indicator_id")
    if value.get("status") == "completed" and "worker_id" in value:
        drop.update(("n_llm_calls", "reused"))
    if value.get("action") == "prepare_data" and "workers" in value:
        drop.update(("action", "raw_data", "model", "simulation_source", "n_observations"))
    if {"model", "panel", "report"} <= value.keys():
        drop.update(("action", "retention"))
    if value.get("action") in ("simulate", "data_diff") and "report" in value:
        drop.update(("action", "panel"))
    if {"design", "latent_paths"} <= value.keys():
        drop.add("assignments")
    return {key: convert_payload(item) for key, item in value.items() if key not in drop}


def current_model_definition(value: JsonObject, loader: ArrayLoader) -> JsonObject:
    """One-time bridge for converter fingerprint parsing and historical layout recovery."""
    if "law_layouts" in value:
        return value
    laws = value["distributions"]
    # Parse native laws at their codec owner without requiring a historical compiler.
    from nof1_causal_lab.numpyro_json import NumPyroDistribution
    from pydantic import TypeAdapter

    parsed = TypeAdapter(dict[str, NumPyroDistribution]).validate_python(
        laws, context={"distribution_array_loader": loader}
    )
    joint = {identity for identity, law in parsed.items() if distribution_shape(law) != ((), ())}

    def uncondition(node):
        if isinstance(node, list):
            return [uncondition(item) for item in node]
        if isinstance(node, Mapping):
            return {
                key: None
                if key == "distribution" and isinstance(item, str) and item in joint
                else uncondition(item)
                for key, item in node.items()
                if key != "time_points"
            }
        return node

    if not joint:
        return {
            key: item for key, item in {**value, "law_layouts": {}}.items() if key != "time_points"
        }
    base = uncondition(value)
    base["distributions"] = {
        identity: law for identity, law in laws.items() if identity not in joint
    }
    model = ModelSpec.model_validate(base, context={"distribution_array_loader": loader})
    layouts = {}
    for identity in sorted(joint):
        parameters = tuple(
            parameter.id
            for parameter in model.parameters
            if next(item for item in value["parameters"] if item["id"] == parameter.id)[
                "distribution"
            ]
            == identity
        )

        # Shared construct definitions may be inline at an edge endpoint.
        def members(node):
            if isinstance(node, list):
                return set().union(*(members(item) for item in node))
            if isinstance(node, Mapping):
                own = (
                    {ConstructId(node["id"])}
                    if "temporal_status" in node and node.get("distribution") == identity
                    else set()
                )
                return own.union(*(members(item) for item in node.values()))
            return set()

        constructs = members(value)
        from nof1_causal_lab.models.model_parameters import execution_parameters
        from nof1_causal_lab.models.ssm.compile import support as numeric
        from nof1_causal_lab.models.ssm.compile.prior_indexing import build_site_bindings
        from nof1_causal_lab.models.ssm.compile.prior_compilation import bind_parameters

        layout = None
        # Only the binding stage is needed. Complete compilation would require
        # the old priors we replaced, and would inspect the law we are labelling.
        for outcome in (None, *sorted(constructs)):
            selection = StructuralSelection(model, outcome)
            dynamics = numeric.dynamics_components(selection)
            states = numeric.state_ids(selection)
            blocks = numeric.parameter_blocks(selection)
            sites = tuple(
                sorted(
                    (
                        *(
                            site
                            for index, component in enumerate(dynamics.components)
                            for site in component.iter_sites(f"vf_{index}", n_latent=len(states))
                        ),
                        *(site for block in blocks for site in block.iter_sites()),
                        *numeric.likelihood_sites(selection),
                    ),
                    key=lambda site: site.name,
                )
            )
            definitions = execution_parameters(selection)
            bindings, _ = bind_parameters(
                build_site_bindings(selection, sites, dynamics.components, definitions),
                selection,
                definitions,
                sites,
            )
            if not set(parameters) <= {binding.parameter_id for binding in bindings}:
                continue
            candidate = joint_law_layout(
                bindings,
                parameters=parameters,
                constructs=constructs,
                time_points=value["time_points"] if constructs else (),
                construct_labels={
                    member: model.get_construct(member).name for member in constructs
                },
            )
            if candidate.distribution_id == identity and distribution_shape(parsed[identity]) == (
                (),
                (candidate.width,),
            ):
                layout = candidate
                break
        if layout is None:
            raise ValueError(
                f"Historical joint law {identity} does not match its recoverable scientific coordinates"
            )
        layouts[identity] = layout.model_dump(mode="json", round_trip=True)
    return {
        key: item for key, item in {**value, "law_layouts": layouts}.items() if key != "time_points"
    }


def migrate(source: Path, destination: Path) -> dict[str, str]:
    """Copy facts, retain leaf topology and remap all stored Git references consistently."""
    repo = pygit2.Repository(str(source / "study/history.git"))
    if repo.config.get_int("nof1.format") != 17:
        raise ValueError("Chain migrate_format_18 after migrate_format_17")
    load = cache(lambda ref: read_array(str(source / "store/arrays"), ref))
    original_reports = {}
    for ref in repo.references:
        if not ref.startswith("refs/attempts/"):
            continue
        commit = repo[repo.references[ref].target].peel(pygit2.Commit)
        record = json.loads(commit.tree["logs/attempt.json"].peel(pygit2.Blob).data)
        attempt = record["attempt"]
        if attempt["action"] == "fit" and attempt["outcome"]["status"] == "applied":
            for info in attempt["outcome"]["effects"]["produced"]:
                if info["artifact_id"] == "model":
                    original_reports[info["revision"]] = attempt["outcome"]["result"]["report"]

    def owned_distribution(outcome, result):
        produced = next(
            info["revision"]
            for info in outcome["effects"]["produced"]
            if info["artifact_id"] == "model"
        )
        fitted = json.loads(
            repo[pygit2.Oid(hex=produced)].peel(pygit2.Tree)["model.json"].peel(pygit2.Blob).data
        )
        prior = json.loads(
            repo[pygit2.Oid(hex=result["model"]["revision"])]
            .peel(pygit2.Tree)["model.json"]
            .peel(pygit2.Blob)
            .data
        )
        layouts = current_model_definition(fitted, load)["law_layouts"]
        changed = tuple(
            identity
            for identity in layouts
            if fitted["distributions"][identity] != prior["distributions"].get(identity)
        )
        if len(changed) != 1:
            raise ValueError(f"Fit must own one changed joint law; found {changed} in {produced}")
        return changed[0]

    def retain(values):
        return (
            write_array(str(destination / "store/arrays"), np.asarray(values))
            if values is not None
            else None
        )

    def update_file(oid: str, name: str, value):
        if name == "model.json":
            definition = current_model_definition(value, load)
            report = original_reports.get(oid)
            if report is not None:
                labels = {
                    item["subject"]["element_id"]: item["parameter"]
                    for item in report["core"]["posterior_marginals"] or ()
                }
                definition = {
                    **definition,
                    "law_layouts": {
                        identity: {
                            **layout,
                            "labels": {
                                element: labels.get(element, label)
                                for element, label in layout["labels"].items()
                            },
                        }
                        for identity, layout in definition["law_layouts"].items()
                    },
                }
            return definition
        if name != "attempt.json":
            return value
        attempt = value["attempt"]
        request = attempt["request"]
        if request is not None and attempt["action"] == "edit_model":
            request = {**request, "model": current_model_definition(request["model"], load)}
        outcome = attempt["outcome"]
        if outcome["status"] != "applied":
            return {**value, "attempt": {**attempt, "request": request}}
        result = outcome["result"]
        if attempt["action"] == "fit":
            if result.get("retention") == "report_only":
                result = None
            else:
                core, detail = result["report"]["core"], result["report"]["detail"]
                diagnostics = core["inference_diagnostics"]
                chains = diagnostics["num_chains"] if diagnostics is not None else None
                diverging = detail.get("divergent")
                extra = (
                    {"diverging": retain(np.asarray(diverging).reshape(chains, -1))}
                    if diverging is not None and chains is not None
                    else {}
                )
                result = {
                    "model": result["model"],
                    "panel": result["panel"],
                    "evidence": {
                        "distribution": owned_distribution(outcome, result),
                        "engine": {}
                        if core["engine"]["kind"] == "evaluated"
                        and core["engine"]["outcome"] == "passed"
                        and convert_payload(core["engine"]["evidence"]) == {}
                        else None,
                        "time_origin": core["time_origin"],
                        "duration_seconds": core["inference_metadata"]["duration_seconds"],
                        "num_chains": chains,
                        "chain_extra_fields": extra,
                        "sampler_diagnostics": convert_payload(core["sampler_diagnostics"]),
                        "initial_latent_delta": retain(detail.get("initial_latent_delta")),
                        "final_latent_delta": retain(detail.get("final_latent_delta")),
                    },
                }
        elif attempt["action"] == "simulate":
            report = result["report"]
            result = {
                "evidence": {
                    key: item
                    for key, item in report.items()
                    if key not in {"findings", "causal", "law", "fit_reliability", "assignments"}
                }
            }
        elif attempt["action"] == "data_diff":
            result = None
        return {
            **value,
            "attempt": {**attempt, "request": request, "outcome": {**outcome, "result": result}},
        }

    return rewrite_study(
        source,
        destination,
        convert_payload,
        update_file=update_file,
        rename_entry=lambda name: (
            None
            if name in _DERIVED
            or name in {"checks.json", "predictive_checks.json", "retained-metadata.json"}
            else name
        ),
        mapping_name="format-18-revisions.json",
        source_format=17,
        target_format=18,
        drop_references=lambda name: any(
            name.startswith(f"refs/artifacts/{artifact}/") for artifact in _DERIVED
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = migrate(args.source.resolve(), args.destination.resolve())
    print(f"Converted format 17 → 18: {len(mapping)} Git objects into {args.destination}")


if __name__ == "__main__":
    main()
