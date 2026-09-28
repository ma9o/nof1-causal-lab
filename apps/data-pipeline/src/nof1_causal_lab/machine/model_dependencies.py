"""Which actual model input each persisted computation consumes.

The revision pins record original input revisions. These input identities permit reuse
on a later model revision when the computation's canonical inputs are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId


MODEL_INPUTS: dict[ArtifactId, str] = {
    "identification_report": "identification",
    "validation_report": "belief",
}
