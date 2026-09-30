"""Migrate a stopped format-4 study to persisted simulation summaries and origins.

Usage: uv run python -m scripts.migrations.migrate_simulation_preparation SOURCE DESTINATION
Only a new copy is written. No extraction, fitting or forward simulation is run.
Imported panels without file recipes require an explicit scientific decision and
are rejected before copying; this migration never invents extraction instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl
import pygit2

from nof1_causal_lab.actions.predictive_checks import law_provenance
from nof1_causal_lab.actions.simulation_summaries import (
    paired_effect_trajectory,
    summarize_simulation,
)
from nof1_causal_lab.artifacts.data_preparation import FileSourceRef, PreparedDataMetadata
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.observations import ObservationSpec
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.machine.artifacts import ArtifactRecord
from nof1_causal_lab.models.ssm.inference.convergence import convergence_failures
from nof1_causal_lab.utils.arrays import read_array
from nof1_causal_lab.utils.data import ensure_datetime_column
from nof1_causal_lab.utils.observation_rows import prepared_time_origin
from nof1_causal_lab.utils.time_coordinates import SYNTHETIC_EPOCH
from scripts.migrations.study_rewrite import remove_edge_timing, rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.machine.store import ArtifactStore


class _Archive:
    def __init__(self, source: Path):
        self.source = source
        self.repo = pygit2.Repository(str(source / "episode/history.git"))
        if self.repo.config.get_int("nof1.format") != 4:
            raise ValueError("Expected a format-4 study")
        self.records = {}
        self.panels = set()
        for name in self.repo.references:
            target = self.repo.references[name].target
            if name.startswith("refs/artifacts/panel/"):
                self.panels.add(str(target))
            if name.startswith(("refs/heads/", "refs/attempts/")):
                for commit in self.repo.walk(target):
                    if "artifacts/panel" in commit.tree:
                        self.panels.add(str(commit.tree["artifacts/panel"].id))
                    if "logs/transition.json" in commit.tree:
                        self.records[str(commit.id)] = self.payload(
                            str(commit.id), "logs/transition.json"
                        )
        unsupported = [
            revision
            for revision in sorted(self.panels)
            if "file" in self.payload(revision, "metadata.json")["source"]
        ]
        if unsupported:
            raise ValueError(
                "Imported panels have no files recipe; decide how to reprepare or archive them before migration: "
                + ", ".join(unsupported)
            )
        if any(
            record.get("operation_id") == "imported_measurements"
            for record in self.records.values()
        ):
            raise ValueError(
                "Retired observation-table attempts remain in this history; decide how to archive them before migration"
            )
        self.fits = {
            item["revision"]: record
            for record in self.records.values()
            if record.get("operation_id") == "posterior" and record["status"] == "applied"
            for item in record["produced"]
            if item["artifact_id"] == "model"
        }

    def payload(self, revision, path):
        tree = self.repo[pygit2.Oid(hex=revision)].peel(pygit2.Tree)
        return json.loads(tree[path].peel(pygit2.Blob).data)

    def read_meta(self, artifact_id, revision):
        del artifact_id
        return ArtifactRecord.model_validate(
            {**self.payload(revision, "meta.json"), "revision": revision}
        )

    def read_json_file(self, artifact_id, revision, filename):
        del artifact_id
        return self.payload(revision, filename)

    def array(self, identity):
        return read_array(str(self.source / "store/arrays"), identity)

    @cache  # noqa: B019 - this archive lives for one offline migration.
    def model(self, revision):
        return ModelSpec.model_validate(
            remove_edge_timing(self.payload(revision, "model.json")),
            context={"distribution_array_loader": self.array},
        )

    @cache  # noqa: B019 - this archive lives for one offline migration.
    def panel(self, revision):
        metadata = self.payload(revision, "metadata.json")
        blob = self.payload(revision, "external.json")["panel.parquet"]
        frame = pl.read_parquet(self.source / "store/blobs" / blob)
        for column in ("anchor_time", "support_start", "support_end"):
            frame = ensure_datetime_column(frame, column)
        source = metadata["source"]
        if "files" in source:
            origin = prepared_time_origin(frame, FileSourceRef.model_validate(source).start)
        else:
            saved = self.simulation(self.records[source["revision"]]["diagnostics"]["report"])
            origin = (
                saved.time_origin + timedelta(days=saved.times[0])
                if saved.time_origin is not None
                else None
            )
            # Old replicates serialized absolute model days from an artificial epoch.
            # Bind known calendar dates; calendar-free replicas instead restart at zero.
            shift = (
                saved.time_origin - SYNTHETIC_EPOCH
                if saved.time_origin is not None
                else -timedelta(days=saved.times[0])
            )
            frame = frame.with_columns(
                pl.col(column) + shift for column in ("anchor_time", "support_start", "support_end")
            )
        return frame, PreparedDataMetadata.model_validate({**metadata, "time_origin": origin})

    def fit_origin(self, record):
        pins = record["diagnostics"]["input_pins"]
        frame, metadata = self.panel(pins["panel"])
        if metadata.time_origin is None:
            return None
        # Historical pivot_to_wide subtracted the earliest selected anchor, even
        # when support boundaries made the retained fit grid start before zero.
        selected = [indicator.id for indicator in self.model(pins["model"]).indicators]
        first = frame.filter(pl.col("indicator_id").is_in(selected))["anchor_time"].min()
        if not isinstance(first, datetime):
            raise ValueError("Historical fit has no selected observation anchor")
        return first.replace(tzinfo=UTC)

    def simulation(self, value):
        revision = GitOid(value["model"]["revision"])
        model = self.model(revision)
        provenance = law_provenance(
            cast("ArtifactStore", self), self.read_meta("model", revision), model, None
        )
        fit = (
            self.fits[provenance.fitted_model_revision]
            if provenance.fitted_model_revision is not None
            else None
        )
        reliability = (
            (
                "unconverged"
                if convergence_failures(fit["diagnostics"]["report"]["inference_diagnostics"])
                else "converged"
            )
            if fit is not None
            else ("unknown" if provenance.kind == "unknown" else "not_fitted")
        )
        action = self.array(value["latent_paths"])
        reference = (
            self.array(value["reference_latent_paths"])
            if value["reference_latent_paths"] is not None
            else None
        )
        layout = value["observation_layout"]
        predictive = summarize_simulation(
            model,
            state_ids=tuple(value["state_ids"]),
            variables=tuple(ObservationSpec.model_validate(item) for item in layout["variables"]),
            latent_paths=action,
            observations=self.array(value["observations"]),
            mask=self.array(layout["mask"]),
            reference_latent_paths=reference,
            reference_observations=self.array(value["reference_observations"])
            if value["reference_observations"] is not None
            else None,
            fit_reliability=reliability,
        )
        causal = value["causal_result"]
        if causal is not None:
            if reference is None:
                raise ValueError("Historical certified effect lacks paired draws")
            index = value["state_ids"].index(causal["outcome"])
            trajectory = paired_effect_trajectory(
                value["times"], action[:, :, index] - reference[:, :, index]
            )
            causal = {
                key: item
                for key, item in causal.items()
                if key not in {"time_grid_days", "trajectories"}
            }
            causal.update(
                effect_trajectory=[point.model_dump(mode="json") for point in trajectory],
                trajectory_peak=max(trajectory, key=lambda point: abs(point.effect)).model_dump(
                    mode="json"
                ),
            )
        return SimulationReport.model_validate(
            {
                **value,
                "time_origin": self.fit_origin(fit) if fit is not None else None,
                "origin_panel_revision": provenance.fitted_panel_revision,
                "predictive": predictive,
                "law": provenance,
                "causal_result": causal,
            }
        )


def migrate_workspace(source: Path, destination: Path) -> dict[str, str]:
    archive = _Archive(source)

    def update(value):
        if isinstance(value, list):
            return [update(item) for item in value]
        if not isinstance(value, dict):
            return value
        if {"observation_layout", "parameter_draws", "times", "design"} <= value.keys():
            return archive.simulation(value).model_dump(mode="json")
        if value.get("operation_id") == "posterior" and value.get("status") == "applied":
            origin = archive.fit_origin(value)
            value = {
                **value,
                "diagnostics": {
                    **value["diagnostics"],
                    "report": {
                        **value["diagnostics"]["report"],
                        "time_origin": origin.isoformat() if origin else None,
                    },
                },
            }
        return {key: update(item) for key, item in value.items()}

    def update_file(revision, filename, value):
        if revision not in archive.panels:
            return value
        frame, metadata = archive.panel(revision)
        if filename == "metadata.json":
            return metadata.model_dump(mode="json")
        if filename == "external.json" and "revision" in metadata.source.model_dump():
            buffer = io.BytesIO()
            frame.write_parquet(buffer)
            payload = buffer.getvalue()
            identity = hashlib.sha256(payload).hexdigest()
            (destination / "store/blobs" / identity).write_bytes(payload)
            return {**value, "panel.parquet": identity}
        return value

    return rewrite_study(
        source,
        destination,
        update,
        update_file=update_file,
        target_format=5,
        mapping_name="simulation-preparation-mapping.json",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = migrate_workspace(args.source, args.destination)
    print(f"Migrated {len(mapping)} Git objects into {args.destination}; source unchanged")


if __name__ == "__main__":
    main()
