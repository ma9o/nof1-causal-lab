"""Record the study question as the first artifact of its lineage."""

from nof1_causal_lab.actions.contracts import SetQuestionRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.store import ArtifactStore


def set_question(workspace_id: str, request: SetQuestionRequest) -> Applied[None]:
    info = ArtifactStore(workspace_id).write_artifact(
        "question",
        derived_from={},
        produced_by="set_question",
        json_files={
            json_filename("question", "question"): request.question.model_dump(mode="json")
        },
    )
    return Applied(result=None, effects=ActionEffects(produced=(info,)))
