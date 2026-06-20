from pydantic import BaseModel, Field, ConfigDict


class ChunkingConfig(BaseModel):
    type: str = Field(description="The chunking type to use.", default="text")
    size: int = Field(description="The chunk size to use", default=1200)
    overlap: int = Field(description="The chunk overlap to use", default=100)
    metadata: list[str] | None = Field(
        description="Metadata fields from the source document", default=None
    )
    model_config = ConfigDict(extra="allow")
