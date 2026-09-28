"""Episode state machine: pure domain layer.

The pipeline is a state machine the navigating LLM traverses freely,
constrained only by artifact-level dependencies. This package holds the
engine-agnostic core:

- :mod:`artifacts` — artifact taxonomy, versions, input metadata
- :mod:`graph` — the artifact-level dependency DAG
- :mod:`hierarchy` — public actions, contexts, execution classes, write effects
- :mod:`execution` — result installation and artifact freshness
- :mod:`errors` — typed transition-execution exceptions

Everything here is pure (no I/O, no engine imports) so it can run inside a
Temporal workflow sandbox, a test, or a notebook unchanged. I/O lives in
:mod:`nof1_causal_lab.machine.store` (versioned artifact store + journal) and
engine wiring lives in :mod:`nof1_causal_lab.machine.temporal`.
"""

from nof1_causal_lab.artifacts.identity import ArtifactId
from nof1_causal_lab.machine.artifacts import ArtifactRecord, EpisodeState
from nof1_causal_lab.machine.errors import (
    ArtifactWriteRejected,
    ModelFitError,
    TransitionExecutionError,
)
from nof1_causal_lab.machine.execution import (
    RetractedArtifact,
    apply_transition,
    freshness_report,
    is_stale,
)
from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    DERIVATIONS,
    ROOTS,
    Derivation,
    Root,
    Transition,
    transition_spec,
)
from nof1_causal_lab.machine.hierarchy import (
    ACTIONS,
    CONTEXTS,
    ActionSpec,
    ContextSpec,
)

__all__ = [
    "ARTIFACT_GRAPH",
    "ACTIONS",
    "ActionSpec",
    "ArtifactId",
    "ArtifactRecord",
    "ArtifactWriteRejected",
    "CONTEXTS",
    "ContextSpec",
    "DERIVATIONS",
    "Derivation",
    "EpisodeState",
    "ModelFitError",
    "ROOTS",
    "Root",
    "RetractedArtifact",
    "Transition",
    "TransitionExecutionError",
    "apply_transition",
    "freshness_report",
    "is_stale",
    "transition_spec",
]
