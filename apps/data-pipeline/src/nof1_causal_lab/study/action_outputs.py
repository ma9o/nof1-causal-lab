"""Read the complete result saved by a completed action, without scientific execution."""

import json

from nof1_causal_lab.actions.call_logs import read_call_log
from nof1_causal_lab.actions.contracts import call_identity
from nof1_causal_lab.actions.results import FailedPoll
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, StudyRevision


def completed_call_json(workspace_id: str, revision: StudyRevision) -> bytes:
    """Return the saved body bytes inside the polling envelope, without rebuilding projections."""
    attempt = revision.record.attempt
    if attempt.request is None:
        raise StudyLookupError("This historical attempt has no retained call arguments")
    repository = StudyRepository(workspace_id)
    identity = call_identity(attempt.request)
    log = read_call_log(repository, revision.commit_id)
    if not isinstance(attempt.outcome, Applied):
        return (
            FailedPoll(
                call_id=identity,
                action=attempt.action,
                commit_id=revision.commit_id,
                messages=log.messages,
            )
            .model_dump_json()
            .encode()
        )
    metadata = {
        "call_id": identity,
        "action": attempt.action,
        "status": "success",
        "commit_id": revision.commit_id,
        "messages": log.model_dump(mode="json")["messages"],
    }
    body = repository.read_file(revision.commit_id, "result.json")
    return json.dumps(metadata, separators=(",", ":")).encode()[:-1] + b',"body":' + body + b"}"
