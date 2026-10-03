"""Rebuild archived fit controls from telemetry and two agreeing Git-pinned defaults.

Usage: uv run python -m scripts.migrations.build_sampler_settings STUDY OUTPUT
The default pins bracket STEPWISE2's 2026-09-29 fits. No current defaults fill gaps.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pygit2

from nof1_causal_lab.sampler_config import MarginalParticleGibbsSpec, SamplerSpec

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonValue

CONFIG_PATH = "apps/data-pipeline/src/nof1_causal_lab/utils/config.py"
DEFAULT_PINS = ("c8d7c3bd", "1a468a4c")


def _pinned_defaults(revision: str) -> dict[str, JsonValue]:
    source = subprocess.run(
        ["git", "show", f"{revision}:{CONFIG_PATH}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "MAPConfig",
            "MarginalParticleGibbsConfig",
            "InferenceConfig",
        }:
            for field in node.body:
                if (
                    isinstance(field, ast.AnnAssign)
                    and isinstance(field.target, ast.Name)
                    and isinstance(field.value, (ast.Constant, ast.Tuple, ast.UnaryOp))
                ):
                    values[field.target.id] = ast.literal_eval(field.value)
    return values


def build_settings(source: Path, pins: tuple[str, str] = DEFAULT_PINS) -> dict[str, JsonValue]:
    """Resolve every field from its recorded owner or agreeing historical config owners."""
    defaults = tuple(_pinned_defaults(pin) for pin in pins)
    repo = pygit2.Repository(str(source / "study/history.git"))
    if repo.config.get_int("nof1.format") != 14:
        raise ValueError("The settings builder requires the original format-14 telemetry")
    result = {}
    for name in sorted(repo.references):
        if not name.startswith("refs/attempts/"):
            continue
        commit = repo.references[name].peel(pygit2.Commit)
        record = json.loads(commit.tree["logs/attempt.json"].peel(pygit2.Blob).data)
        attempt = record["attempt"]
        if attempt["action"] != "fit" or attempt["outcome"]["status"] != "applied":
            continue
        request = attempt["request"]
        if request is None or any(value is not None for value in request["settings"].values()):
            raise ValueError("This rebuild requires retained requests with no sampler overrides")
        sampler = attempt["outcome"]["result"]["report"]["sampler_diagnostics"]
        initialization = sampler["initialization"]
        warmup = sampler["parameter_warmup"]
        preconditioner = sampler["preconditioner"]
        pathfinder = initialization["pathfinder"]
        controls = {
            key: sampler[key] for key in MarginalParticleGibbsSpec.model_fields if key in sampler
        }
        controls.update(
            amala_grad_clip="infinity"
            if sampler["amala_grad_clip"] is None
            else sampler["amala_grad_clip"],
            param_step_size=sampler["param_step_size_initial"],
            init_method=initialization["init_method"],
            init_scale=warmup["init_scale"],
            pathfinder_init_scale=warmup["pathfinder_init_scale"],
            auto_preconditioner_method=preconditioner["auto_preconditioner_method"],
        )
        if (
            warmup["pathfinder_init_scale"] != initialization["pathfinder_init_scale"]
            or warmup["auto_preconditioner_method"] != preconditioner["auto_preconditioner_method"]
        ):
            raise ValueError("Conflicting retained initialization/preconditioner controls")
        if pathfinder is not None:
            for setting, measurement in (
                ("pathfinder_num_elbo_samples", "pathfinder_elbo_samples"),
                ("pathfinder_maxiter", "pathfinder_maxiter"),
                ("n_pathfinder_starts", "n_pathfinder_starts"),
                ("pathfinder_parallel_workers", "pathfinder_parallel_workers"),
            ):
                controls[setting] = pathfinder[measurement]
        if preconditioner["auto_preconditioner_maxiter"] is not None:
            controls["auto_preconditioner_maxiter"] = preconditioner["auto_preconditioner_maxiter"]
        common = {key: sampler[key] for key in SamplerSpec.model_fields if key in sampler}
        pinned = {}
        for fields, values in (
            (set(SamplerSpec.model_fields) - {"marginal_particle_gibbs"}, common),
            (set(MarginalParticleGibbsSpec.model_fields), controls),
        ):
            for field in sorted(fields - values.keys()):
                if (
                    any(field not in owner for owner in defaults)
                    or defaults[0][field] != defaults[1][field]
                ):
                    raise ValueError(f"Historical defaults are missing or disagree for {field}")
                values[field] = defaults[0][field]
                pinned[field] = defaults[0][field]
        settings = SamplerSpec.model_validate({**common, "marginal_particle_gibbs": controls})
        assert settings.model_fields_set == set(SamplerSpec.model_fields)
        assert settings.marginal_particle_gibbs.model_fields_set == set(
            MarginalParticleGibbsSpec.model_fields
        )
        result[str(commit.id)] = settings.model_dump(mode="json", round_trip=True)
        print(
            "Git-pinned sampler fields: "
            + json.dumps(
                {
                    "commit": str(commit.id),
                    "seq": record["seq"],
                    "pins": list(pins),
                    "fields": pinned,
                    "recorded_fields": sorted((set(common) | set(controls)) - pinned.keys()),
                },
                sort_keys=True,
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    settings = build_settings(args.source.resolve())
    args.output.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    print(f"Rebuilt full SamplerSpec controls for {len(settings)} archived fits")


if __name__ == "__main__":
    main()
