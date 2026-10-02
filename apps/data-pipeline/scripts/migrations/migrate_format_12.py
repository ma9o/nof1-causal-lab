"""Convert stopped format-11 studies to typed scientific alternatives in format 12.

Usage: uv run python -m scripts.migrations.migrate_format_12 SOURCE DESTINATION
Only the new destination is written. No numerical values are regenerated.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pygit2
from pydantic import TypeAdapter

from nof1_causal_lab.actions.model_checks import CHECK_POLICY_VERSION
from nof1_causal_lab.actions.predictive_checks import PREDICTIVE_POLICY_VERSION
from nof1_causal_lab.artifacts.identity import scientific_id
from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonValue

_OLD_LAWS = {
    "Delta",
    "Normal",
    "StudentT",
    "Poisson",
    "Gamma",
    "Bernoulli",
    "NegativeBinomial2",
    "Beta",
    "OrderedLogistic",
    "Categorical",
}
_INTERVAL_TRANSFORMS = {"dt_persistence_to_ct_decay", "dt_effect_to_ct_rate"}


def convert_payload(value: JsonValue) -> JsonValue:
    """Translate old owned records wherever nested, retaining null payloads as values."""
    if isinstance(value, (list, tuple)):
        return [convert_payload(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    converted = {key: convert_payload(item) for key, item in value.items()}
    distribution = converted.get("distribution")
    if isinstance(distribution, str) and distribution in _OLD_LAWS and "arguments" in converted:
        arguments = converted.pop("arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError("Format-11 observation arguments must be an object")
        if distribution == "Bernoulli":
            if set(arguments) == {"logits"}:
                converted["distribution"] = "BernoulliLogits"
            elif set(arguments) == {"probs"}:
                converted["distribution"] = "BernoulliProbs"
            else:
                raise ValueError(
                    "Bernoulli must retain exactly its logits or probs parameterization"
                )
        converted.update(arguments)

    identity = converted.get("id")
    if isinstance(identity, str) and identity.startswith("parameter:") and "name" in converted:
        transform = converted.pop("distribution_transform", "identity")
        interval = converted.pop("reference_interval_days", None)
        if not isinstance(transform, str):
            raise ValueError("Format-11 parameter transform must be a tag")
        if transform in _INTERVAL_TRANSFORMS:
            converted["transform"] = {
                "kind": transform,
                "interval_days": "model_clock" if interval is None else interval,
            }
        else:
            if interval is not None:
                raise ValueError(
                    "A non-interval transform carries an unexplained interval; audit it"
                )
            converted["transform"] = {"kind": transform}
    if (
        isinstance(identity, str)
        and identity.startswith("mechanism:")
        and "expression" in converted
    ):
        converted.setdefault("kind", "drift")

    kind = converted.get("kind")
    if kind == "panel" and "revision" in converted and converted.pop("replicate", None) is not None:
        raise ValueError("A panel cannot carry a simulation replicate")
    if kind in ("authored", "unknown") and "interpretation" in converted:
        for field in ("fitted_panel_revision", "fitted_model_revision"):
            if converted.pop(field, None) is not None:
                raise ValueError("Non-fitted provenance contains an unexplained fitted reference")
    if "worker_id" in converted and "n_windows" in converted:
        if converted.get("status") == "completed":
            if converted.pop("error", None) is not None:
                raise ValueError("A completed extraction retains an unexplained failure")
        elif converted.get("status") == "failed" and converted.pop("result_ref", None) is not None:
            raise ValueError("A failed extraction retains an unexplained result")

    change = converted.get("change")
    if isinstance(change, str) and change in {"added", "removed", "revised", "unchanged"}:
        if "construct_id" in converted or "edge_id" in converted:
            before, after = converted.pop("before"), converted.pop("after")
        elif "anchor_time" in converted:
            before, after = converted.pop("left"), converted.pop("right")
        elif "parameter_id" in converted or "path" in converted:
            before, after = converted.pop("before"), converted.pop("after")
        else:
            return converted
        replacement: dict[str, JsonValue] = {"kind": change}
        if change != "added":
            replacement["before"] = before
        if change != "removed":
            replacement["after"] = after
        converted["change"] = replacement
    return converted


def convert_study(source: Path, destination: Path) -> dict[str, str]:
    return rewrite_study(
        source,
        destination,
        convert_payload,
        mapping_name="format-12-revisions.json",
        source_format=11,
        target_format=12,
        preserve_model_meaning=True,
        check_preimages=_check_preimages(source),
    )


def _check_preimages(source: Path) -> dict[str, JsonValue]:
    """Recognize retained check inputs without executing checks or refreshing stale evidence."""
    repo = pygit2.Repository(str(source / "study/history.git"))
    preimages: dict[str, JsonValue] = {}

    def fingerprints(oid: str) -> dict[str, str]:
        metadata = json.loads(repo[oid].peel(pygit2.Tree)["meta.json"].peel(pygit2.Blob).data)
        return TypeAdapter(dict[str, str]).validate_python(metadata["model_inputs"])

    def remember(group: str, values: JsonValue) -> None:
        values = [CHECK_POLICY_VERSION, group, values]
        preimages[scientific_id("check", values)] = values

    def predictive(value: JsonValue) -> None:
        if isinstance(value, Mapping):
            if "input_key" in value and "model_revision" in value and "law" in value:
                inputs = fingerprints(str(value["model_revision"]))
                for compatible in (False, True):
                    candidate = [
                        PREDICTIVE_POLICY_VERSION,
                        inputs["compilation"],
                        inputs["belief"],
                        value["panel_revision"],
                        compatible,
                        value["draws"],
                        value["seed"],
                        value["law"],
                    ]
                    if scientific_id("check", candidate) == value["input_key"]:
                        preimages[str(value["input_key"])] = candidate
            for child in value.values():
                predictive(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                predictive(child)

    visited: set[str] = set()
    for name in repo.references:
        if name.startswith("refs/artifacts/model/"):
            inputs = fingerprints(str(repo.references[name].target))
            remember("specification", [inputs["compilation"], inputs["belief"]])
            remember("identification", inputs["identification"])
        if not name.startswith(("refs/heads/", "refs/attempts/")):
            continue
        for commit in repo.walk(repo.references[name].target):
            if str(commit.id) in visited:
                continue
            visited.add(str(commit.id))
            if "artifacts/model" in commit.tree:
                inputs = fingerprints(str(commit.tree["artifacts/model"].id))
                remember("specification", [inputs["compilation"], inputs["belief"]])
                remember("identification", inputs["identification"])
                if "artifacts/panel" in commit.tree and "artifacts/data_profile" in commit.tree:
                    remember(
                        "compatibility",
                        [
                            inputs["observations"],
                            inputs["belief"],
                            str(commit.tree["artifacts/panel"].id),
                            str(commit.tree["artifacts/data_profile"].id),
                        ],
                    )
            for filename in ("checks.json", "logs/attempt.json"):
                if filename in commit.tree:
                    predictive(json.loads(commit.tree[filename].peel(pygit2.Blob).data))
    return preimages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = convert_study(args.source, args.destination)
    print(f"Converted format 11 → 12: {len(mapping)} mapped Git objects")


if __name__ == "__main__":
    main()
