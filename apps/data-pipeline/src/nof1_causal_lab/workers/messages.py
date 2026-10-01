"""Production message builders for measurement-extraction subroutines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.utils.observation_semantics import get_observation_semantics
from nof1_causal_lab.workers.prompts.extraction import SYSTEM, USER

if TYPE_CHECKING:
    from nof1_causal_lab.workers.context import MeasurementContext


def _format_indicators(measurement_structure: MeasurementContext) -> str:
    """Format indicators and their observation semantics for a worker prompt."""
    lines = []
    model_clock = measurement_structure.get("model_clock", "")
    for indicator in measurement_structure.get("indicators", []):
        name = indicator.get("name", "unknown")
        how_to_measure = indicator.get("how_to_measure", "")
        dtype = indicator.get("measurement_dtype", "")
        semantics = get_observation_semantics(indicator)
        support_kind = indicator.get("support_kind") or semantics.support_kind.value
        summary_operator = indicator.get("summary_operator") or semantics.summary_operator.value
        window = indicator.get("observation_window") or model_clock
        ordinal_levels = indicator.get("ordinal_levels") or []

        details = [dtype, f"operator={summary_operator}", f"support={support_kind}"]
        if window:
            details.append(f"window={window}")
        levels = ordinal_levels if dtype == "ordinal" else indicator.get("categorical_levels")
        if dtype in {"ordinal", "categorical"} and levels:
            codebook = ", ".join(f"{index}={level}" for index, level in enumerate(levels))
            details.append(f"{dtype}_codes={codebook}")

        lines.append(f"- {name} [{indicator['id']}] ({', '.join(details)}): {how_to_measure}")
    return "\n".join(lines)


@dataclass
class WorkerMessages:
    """Build the prompt messages for one measurement-extraction chunk."""

    question: str
    measurement_structure: MeasurementContext
    window_text: str
    n_windows: int

    def extraction_messages(self) -> list[dict[str, str]]:
        indicators_text = _format_indicators(self.measurement_structure)
        return [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER.format(
                    question=self.question,
                    indicators=indicators_text,
                    n_windows=self.n_windows,
                    window_text=self.window_text,
                ),
            },
        ]


__all__ = ["WorkerMessages"]
