"""The complete result of a successful action is its durable cache entry."""

from nof1_causal_lab.actions.io import (
    DataDiffOutput,
    EditModelOutput,
    EditQuestionOutput,
    FitOutput,
    ModelDiffOutput,
    PrepareDataOutput,
    SimulateOutput,
)
from nof1_causal_lab.artifacts.identity import ActionId, GitOid
from nof1_causal_lab.study.store import ArtifactStore

type ActionOutput = (
    EditQuestionOutput
    | EditModelOutput
    | PrepareDataOutput
    | FitOutput
    | SimulateOutput
    | DataDiffOutput
    | ModelDiffOutput
)


def read_result(store: ArtifactStore, action: ActionId, revision: GitOid) -> ActionOutput:
    """Decode the saved result with the schema owned by its recorded action."""
    match action:
        case "edit_question":
            return store.read_report(revision, EditQuestionOutput)
        case "edit_model":
            return store.read_report(revision, EditModelOutput)
        case "prepare_data":
            return store.read_report(revision, PrepareDataOutput)
        case "fit":
            return store.read_report(revision, FitOutput)
        case "simulate":
            return store.read_report(revision, SimulateOutput)
        case "data_diff":
            return store.read_report(revision, DataDiffOutput)
        case "model_diff":
            return store.read_report(revision, ModelDiffOutput)
