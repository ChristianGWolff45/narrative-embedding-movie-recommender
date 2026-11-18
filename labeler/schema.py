from pydantic import BaseModel, Field


class MovieLabel(BaseModel):
    themes: list[str] = Field(default_factory=list)
    tone: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    narrative_structure: list[str] = Field(default_factory=list)
    rationale: str = ""
    model_name: str = ""
    prompt_version: str = "v1"
