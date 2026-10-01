"""Storage failures shared by persistence and the action executor."""


class ArtifactWriteRejected(ValueError):
    """A model edit's payload failed schema validation."""

    def __init__(self, message: str, *, artifact_id: str) -> None:
        super().__init__(message)
        self.artifact_id = artifact_id
