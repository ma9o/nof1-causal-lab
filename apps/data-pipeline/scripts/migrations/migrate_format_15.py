"""Translate stopped format-14 studies into the composed format-15 contracts.

Usage: uv run python -m scripts.migrations.migrate_format_15 SOURCE DESTINATION
Only a new destination is written. Draws and certification are retained verbatim;
no inference, prediction or causal certification runs during conversion.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pygit2

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior_diagnostics import ParticleSamplerDiagnostics
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model
from nof1_causal_lab.models.ssm.observation_support import augment_wide_data_with_support_boundaries
from nof1_causal_lab.models.ssm.runtime import project_observation_data
from nof1_causal_lab.sampler_config import SamplerSpec
from nof1_causal_lab.utils.arrays import read_array
from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonValue


def _convert_sampler(record: Mapping[str, JsonValue]) -> JsonValue:
    """Fold old measured records using verified original controls, never current defaults."""
    settings = record.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError(
            "A format-14 sampler report needs its original fully resolved SamplerSpec; provide --sampler-settings keyed by the fit commit"
        )
    initialization, preconditioner, old_warmup = (
        record[key] for key in ("initialization", "preconditioner", "parameter_warmup")
    )
    assert isinstance(initialization, Mapping)
    assert isinstance(preconditioner, Mapping)
    assert isinstance(old_warmup, Mapping)
    parsed_settings = SamplerSpec.model_validate(settings)
    configured = parsed_settings.model_dump(mode="json")["marginal_particle_gibbs"]
    assert isinstance(configured, Mapping)
    for name in ("num_warmup", "num_samples", "num_chains", "n_particles"):
        if record[name] != getattr(parsed_settings, name):
            raise ValueError(f"Original sampler controls disagree with retained evidence: {name}")
    for name in (
        "latent_delta",
        "n_parameter_particles",
        "parameter_proposal",
        "amala_delta_init",
        "amala_delta_min",
        "amala_delta_max",
        "amala_target_accept",
        "amala_adaptation_window",
        "amala_adaptation_tolerance",
        "amala_adaptation_rho",
        "amala_adaptation_rho_min",
        "amala_adaptation_gamma",
        "amala_kappa",
        "diagnostic_metrics_all",
        "param_step_size_min",
        "param_step_size_max",
        "adaptation_scheme",
        "latent_init_method",
    ):
        if record[name] != configured[name]:
            raise ValueError(f"Original sampler controls disagree with retained evidence: {name}")
    if (
        record["param_step_size_initial"] != configured["param_step_size"]
        or old_warmup["init_scale"] != configured["init_scale"]
    ):
        raise ValueError("Original initialization scales disagree with retained evidence")
    clip = record["amala_grad_clip"]
    if ("infinity" if clip is None else clip) != configured["amala_grad_clip"]:
        raise ValueError("Original gradient clip disagrees with retained evidence")
    pathfinder = initialization["pathfinder"]
    if isinstance(pathfinder, Mapping):
        pathfinder = dict(pathfinder)
        if pathfinder.pop("pathfinder_elbo") != pathfinder["best_pathfinder_elbo"]:
            raise ValueError("Conflicting Pathfinder ELBO measurements cannot be folded")
        for key in (
            "pathfinder_setup_seconds",
            "pathfinder_jax_compile_seconds",
            "pathfinder_runtime_seconds",
            "pathfinder_total_seconds",
            "pathfinder_jax_compile_batch_sizes",
        ):
            if old_warmup[key] is not None and old_warmup[key] != pathfinder[key]:
                raise ValueError(f"Conflicting duplicate Pathfinder evidence: {key}")
        for old, native in (
            ("auto_preconditioner_n_pathfinder_starts", "n_pathfinder_starts"),
            ("auto_preconditioner_n_pathfinder_starts_finite", "n_pathfinder_starts_finite"),
            ("auto_preconditioner_best_pathfinder_elbo", "best_pathfinder_elbo"),
            ("auto_preconditioner_pathfinder_elbo_spread", "pathfinder_elbo_spread"),
        ):
            if preconditioner[old] is not None and preconditioner[old] != pathfinder[native]:
                raise ValueError(f"Conflicting duplicate Pathfinder evidence: {old}")
    if bool(old_warmup["pathfinder_ran"]) != (pathfinder is not None):
        raise ValueError("The original report does not retain its executed Pathfinder result")
    warmup = {
        name: old_warmup[name]
        for name in (
            "pathfinder_run_count",
            "pathfinder_consumers",
            "init_source",
            "preconditioner_source",
            "dim",
            "duration_seconds",
            "pathfinder_init_scale",
        )
    }
    warmup.update(
        pathfinder=pathfinder,
        preconditioner_device=preconditioner["auto_preconditioner_device"],
        pathfinder_sampling_mode=initialization["pathfinder_sampling_mode"],
        prior_released_site_names=initialization["prior_released_site_names"] or [],
        prior_released_site_indices=initialization["prior_released_site_indices"] or [],
        prior_release_scale=initialization["prior_release_scale"]
        if initialization["prior_release_scale"] is not None
        else 0.0,
    )
    result = {
        name: record[name]
        for name in ParticleSamplerDiagnostics.model_fields
        if name not in {"settings", "parameter_warmup"}
    }
    result.update(settings=settings, parameter_warmup=warmup)
    return ParticleSamplerDiagnostics.model_validate(result).model_dump(
        mode="json", round_trip=True
    )


def convert_payload(value: JsonValue) -> JsonValue:
    """Translate records after overlay schedules have been bound at the edge."""
    if isinstance(value, (list, tuple)):
        return [convert_payload(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    record = {key: convert_payload(item) for key, item in value.items()}
    if record.get("status") == "applied" and isinstance(record.get("result"), Mapping):
        old_result = record["result"]
        assert isinstance(old_result, Mapping)
        result = dict(old_result)
        effects = {key: result.pop(key) for key in ("produced", "retracted", "checks")}
        return {
            "status": "applied",
            "result": None if result["action"] in {"edit_model", "set_question"} else result,
            "effects": effects,
        }
    if set(record) == {"findings"}:
        return record["findings"]
    if "x_values" in record and "density" in record:
        curve = {"x": record.pop("x_values"), "density": record.pop("density")}
        return {**record, "density_curve": curve} if "subject" in record else curve
    if {"scope", "assessments", "checked", "messages", "status"} <= record.keys():
        return {"scope": record["scope"], "assessments": record["assessments"]}
    if {"input_key", "model_revision", "status", "law"} <= record.keys():
        status = record.pop("status")
        findings, checks = record.pop("findings"), record.pop("predictive_checks")
        reason, detail = record.pop("reason"), record.pop("detail")
        record["evaluation"] = (
            {"kind": "unavailable", "reason": reason, "detail": detail}
            if status == "not_evaluated"
            else {"kind": "evaluated", "findings": findings, "predictive_checks": checks}
        )
    if {"inference_metadata", "convergence", "detail"} <= record.keys():
        detail = record.pop("detail")
        assert isinstance(detail, dict)
        pairs = detail["posterior_pairs"]
        assert pairs is None or isinstance(pairs, list)
        old_pairs = []
        for pair in pairs or []:
            assert isinstance(pair, Mapping)
            old_pairs.append(pair)
        flags = old_pairs[0]["divergent"] if old_pairs else None
        if any(pair["divergent"] != flags for pair in old_pairs):
            raise ValueError("Pair divergence flags differ; the original evidence cannot be folded")
        detail["divergent"] = flags
        detail["posterior_pairs"] = (
            [[pair["subject_x"], pair["subject_y"]] for pair in old_pairs]
            if pairs is not None
            else None
        )
        return {"core": record, "detail": detail}
    if {"model", "design", "latent_paths", "predictive"} <= record.keys():
        predictive = record.pop("predictive")
        assert isinstance(predictive, Mapping)
        record["fit_reliability"] = predictive["fit_reliability"]
    if {"outcome", "summary", "effect_trajectory", "warnings"} <= record.keys():
        return {key: record[key] for key in ("outcome", "labels", "warnings")}
    if {"id", "type", "function"} <= record.keys():
        function = record.pop("function")
        assert isinstance(function, Mapping)
        return {**record, "name": function["name"], "arguments": function["arguments"]}
    if set(record) in ({"parameter_id", "change"}, {"anchor_time", "change"}):
        return record["change"]
    if {"parameter_warmup", "initialization", "preconditioner", "latent_kernel"} <= record.keys():
        return _convert_sampler(record)
    return record


def _preflight_overlays(checks, reconstruct, owner) -> bool:
    """Drop only uncompileable evaluations; a reconstructed mismatch is an error."""
    if not checks["overlays"]:
        return False
    commit, seq, attempt_id, _scope = owner
    try:
        times, _scales = reconstruct()
    except (AggregatedCompileError, IncompleteModelError) as exc:
        for overlay in checks["overlays"]:
            print(
                "Dropped archived overlay: "
                + json.dumps(
                    {
                        "commit": commit,
                        "seq": seq,
                        "attempt_id": attempt_id,
                        "indicator": overlay["indicator_id"],
                        "stored_points": len(overlay["observed"]),
                        "reason": str(exc),
                    }
                )
            )
        return True
    failures = [
        {
            "commit": commit,
            "seq": seq,
            "attempt_id": attempt_id,
            "indicator": overlay["indicator_id"],
            "stored_points": len(overlay["observed"]),
            "reconstructed_points": len(times),
        }
        for overlay in checks["overlays"]
        if len(times) != len(overlay["observed"])
    ]
    if failures:
        raise ValueError("Archived overlay conversion stopped:\n" + json.dumps(failures, indent=2))
    return False


def convert_study(
    source: Path, destination: Path, *, sampler_settings: Mapping[str, JsonValue] | None = None
) -> dict[str, str]:
    repo = pygit2.Repository(str(source / "study/history.git"))
    commits = {
        str(commit.id): commit
        for name in repo.references
        if name.startswith(("refs/heads/", "refs/attempts/", "refs/actions/"))
        for commit in repo.walk(repo.references[name].target)
    }
    archived: dict[str, tuple[str, str, JsonValue, str | None]] = {}
    owners = {}
    fitted_origins: dict[str, JsonValue] = {}
    settings_by_logs = {}
    for commit in commits.values():
        if "logs/attempt.json" not in commit.tree:
            continue
        record = json.loads(commit.tree["logs/attempt.json"].peel(pygit2.Blob).data)
        attempt = record["attempt"]
        question = (
            QuestionSpec.model_validate_json(
                commit.tree["artifacts/question/question.json"].peel(pygit2.Blob).data
            )
            if "artifacts/question/question.json" in commit.tree
            else None
        )
        # Historical report-only fits predate the question-rooted action contract.
        outcome = (
            None
            if attempt["request"] is None
            else question.outcome
            if question is not None
            else None
        )
        owners[str(commit.tree["logs"].id)] = (
            str(commit.id),
            record["seq"],
            record["attempt_id"],
            outcome,
        )
        if attempt["action"] != "fit" or attempt["outcome"]["status"] != "applied":
            continue
        result = attempt["outcome"]["result"]
        origin = result["report"]["time_origin"]
        if sampler_settings is not None and str(commit.id) in sampler_settings:
            settings = SamplerSpec.model_validate(sampler_settings[str(commit.id)])
            if settings.model_fields_set != set(
                SamplerSpec.model_fields
            ) or settings.marginal_particle_gibbs.model_fields_set != set(
                type(settings.marginal_particle_gibbs).model_fields
            ):
                raise ValueError(
                    "Provide every original resolved SamplerSpec field; defaults are not migration evidence"
                )
            settings_by_logs[str(commit.tree["logs"].id)] = settings.model_dump(mode="json")
        for info in result["produced"]:
            if info["artifact_id"] == "model":
                fitted_origins[info["revision"]] = origin
        archived[str(commit.tree["logs"].id)] = (
            result["model"]["revision"],
            result["panel"]["revision"],
            origin,
            outcome,
        )

    @cache
    def schedule(
        model_revision: str, panel_revision: str, origin_json: str | None, outcome: str | None
    ):
        from datetime import datetime

        model_tree = repo[pygit2.Oid(hex=model_revision)].peel(pygit2.Tree)
        model = ModelSpec.model_validate_json(
            model_tree["model.json"].peel(pygit2.Blob).data,
            context={
                "distribution_array_loader": lambda key: read_array(
                    str(source / "store/arrays"), key
                )
            },
        )
        panel_tree = repo[pygit2.Oid(hex=panel_revision)].peel(pygit2.Tree)
        files = json.loads(panel_tree["external.json"].peel(pygit2.Blob).data)
        panel = pl.read_parquet(source / "store/blobs" / files["panel.parquet"])
        origin = datetime.fromisoformat(origin_json) if origin_json is not None else None
        from nof1_causal_lab.artifacts.identity import ConstructId

        selection = StructuralSelection(
            model, ConstructId(outcome) if outcome is not None else None
        )
        projected = project_observation_data(
            panel, model_spec=compile_executable_model(selection), time_origin=origin
        )
        from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure

        if isinstance(projected, ObservationPreflightFailure):
            raise ValueError(f"Archived evaluation projection failed: {projected.message}")
        wide, rows = projected
        wide = augment_wide_data_with_support_boundaries(rows, wide, time_origin=origin)
        return tuple(float(value) for value in wide["time"]), {
            indicator.observation.id: law.standardized
            for indicator, law in model.iter_likelihoods()
        }

    def bind_overlays(value, context=None, outcome=None):
        if isinstance(value, list):
            return [bind_overlays(item, context, outcome) for item in value]
        if not isinstance(value, dict):
            return value
        if {"input_key", "model_revision", "panel_revision", "law"} <= value.keys() and value[
            "panel_revision"
        ] is not None:
            panel = repo[pygit2.Oid(hex=value["panel_revision"])].peel(pygit2.Tree)
            origin = json.loads(panel["metadata.json"].peel(pygit2.Blob).data)["time_origin"]
            law = value["law"]
            if law is not None and law["kind"] in {"fitted", "mixed"}:
                origin = fitted_origins[law["fitted_model_revision"]]
            context = value["model_revision"], value["panel_revision"], origin, outcome
        converted = {key: bind_overlays(item, context, outcome) for key, item in value.items()}
        if {"indicator_id", "observed", "median", "spaghetti_draws"} <= value.keys():
            if context is None:
                raise ValueError(
                    "Predictive overlay has no retained source for its evaluated schedule"
                )
            model, panel, origin, scope = context
            times, scales = schedule(model, panel, origin, scope)
            converted.update(
                times=list(times), time_origin=origin, standardized=scales[value["indicator_id"]]
            )
            if len(times) != len(value["observed"]):
                raise ValueError("Archived overlay and pinned evaluation schedule differ")
        return converted

    def bind_settings(value, settings):
        if isinstance(value, list):
            return [bind_settings(item, settings) for item in value]
        if not isinstance(value, dict):
            return value
        bound = {key: bind_settings(item, settings) for key, item in value.items()}
        if {
            "parameter_warmup",
            "initialization",
            "preconditioner",
            "latent_kernel",
        } <= value.keys() and settings is not None:
            bound["settings"] = settings
        return bound

    def update_file(oid, name, value):
        if name == "predictive_checks.json" and oid in dropped_logs:
            value = {**value, "overlays": []}
        context = archived.get(oid) if name == "predictive_checks.json" else None
        bound = bind_overlays(value, context, owners.get(oid, (None, None, None, None))[3])
        return bind_settings(bound, settings_by_logs.get(oid))

    # The approved archive exception removes overlays, retaining all other evidence.
    # A compiled schedule mismatch still stops before writing any destination.
    dropped_logs = set()
    for oid, context in archived.items():
        tree = repo[pygit2.Oid(hex=oid)].peel(pygit2.Tree)
        if "predictive_checks.json" not in tree:
            continue
        checks = json.loads(tree["predictive_checks.json"].peel(pygit2.Blob).data)
        if _preflight_overlays(checks, partial(schedule, *context), owners[oid]):
            dropped_logs.add(oid)

    return rewrite_study(
        source,
        destination,
        convert_payload,
        update_file=update_file,
        mapping_name="format-15-revisions.json",
        source_format=14,
        target_format=15,
        preserve_model_meaning=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--sampler-settings",
        type=Path,
        help="Verified original full SamplerSpec values, keyed by fit commit, for format-14 reports missing resolved controls",
    )
    args = parser.parse_args()
    mapping = convert_study(
        args.source.resolve(),
        args.destination.resolve(),
        sampler_settings=json.loads(args.sampler_settings.read_text())
        if args.sampler_settings
        else None,
    )
    print(f"Converted format 14 → 15: {len(mapping)} mapped Git objects")


if __name__ == "__main__":
    main()
