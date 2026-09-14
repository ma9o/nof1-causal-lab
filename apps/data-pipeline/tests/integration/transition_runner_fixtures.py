"""Small canonical artifacts for fixture-backed runner contract tests."""

from datetime import datetime, timedelta

import polars as pl

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.machine.artifacts import EpisodeState
from tests.helpers import complete_test_model, fixture_entity_id, make_model


def scientific_model():
    model = make_model(["Stress", "Sleep"])
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                c.model_copy(
                    update={
                        "indicators": (
                            c.indicators[0].model_copy(
                                update={
                                    "id": fixture_entity_id("indicator", name),
                                    "name": name,
                                }
                            ),
                        )
                    }
                )
                for c, name in zip(
                    (model.get_construct(identity) for identity in model.state_order),
                    ["stress_score", "sleep_score"],
                    strict=True,
                )
            ),
        )
    )
    return complete_test_model(model)


def panel_frame(n_days=20):
    start = datetime(2024, 1, 1)
    return pl.DataFrame(
        [
            {
                "indicator_id": fixture_entity_id("indicator", indicator),
                "value": value,
                "anchor_time": (start + timedelta(days=day + 1)).isoformat(),
                "support_start": (start + timedelta(days=day)).isoformat(),
                "support_end": (start + timedelta(days=day + 1)).isoformat(),
                "support_kind": "interval",
                "summary_operator": "mean",
                "anchor_policy": "support_end",
                "observation_window": "1d",
            }
            for day in range(n_days)
            for indicator, value in (
                ("stress_score", float(1 + day % 5)),
                ("sleep_score", float(8 - day % 4)),
            )
        ]
    )


def state_from(*infos):
    return EpisodeState().with_versions(list(infos))


def seed_model(store):
    return store.write_version(
        "model",
        provenance="llm",
        derived_from={},
        produced_by="run:statistical_model_spec",
        json_files={"model.json": scientific_model().model_dump(mode="json")},
    )


def seed_panel(store, *, model_version=1):
    return store.write_version(
        "panel",
        provenance="computed",
        derived_from={"model": model_version},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": panel_frame()},
    )
