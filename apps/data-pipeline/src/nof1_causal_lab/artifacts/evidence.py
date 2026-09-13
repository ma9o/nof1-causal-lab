"""Literature evidence shared by authored scientific model decisions."""

from pydantic import BaseModel, Field


class LiteratureSource(BaseModel):
    """A literature source records cited evidence supporting a scientific modeling decision."""

    title: str = Field(description="Title of the source (paper, meta-analysis, textbook, etc.)")
    url: str | None = Field(default=None, description="URL of the source if available")
    snippet: str = Field(description="Relevant excerpt or paraphrase from the source")
