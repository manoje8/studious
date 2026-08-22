from pydantic import BaseModel, ConfigDict, Field


class ChunkingConfig(BaseModel):
    type: str = Field(description="The chunking type to use.", default="text")
    size: int = Field(description="The chunk size to use", default=1200)
    overlap: int = Field(description="The chunk overlap to use", default=100)
    size_mode: str = Field(
        description="How `size` is measured: 'characters' or 'tokens'.", default="tokens"
    )
    metadata: list[str] | None = Field(
        description="Metadata fields from the source document", default=None
    )
    model_config = ConfigDict(extra="allow")
