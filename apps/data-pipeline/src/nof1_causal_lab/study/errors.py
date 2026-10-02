"""Storage failures shared by persistence and the action executor."""


class ArtifactWriteRejected(Exception):
    """A model edit's payload failed schema validation."""

    def __init__(self, message: str, *, artifact_id: str) -> None:
        super().__init__(message)
        self.artifact_id = artifact_id


class StudyLookupError(Exception):
    """A requested branch, revision, artifact, or stored path does not exist."""
