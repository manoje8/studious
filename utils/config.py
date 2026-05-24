import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Config(BaseSettings):
    PROJECT_ID: str = os.getenv("PROJECT_ID")
    LOCATION: str = os.getenv("")
    GCP_DOC_AI_LOCATION: str = os.getenv("GCP_DOC_AI_LOCATION")
    GCP_DOC_AI_PROCESSOR_ID: str = os.getenv("GCP_DOC_AI_PROCESSOR_ID")
    GCP_RAW_BUCKET: str = os.getenv("GCP_RAW_BUCKET")
    GCP_PROCESSED_BUCKET: str = os.getenv("GCP_PROCESSED_BUCKET")
    VPC_CONNECTOR: str = os.getenv("VPC_CONNECTOR")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT: str = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    LOGFIRE_TOKEN: str = os.getenv("LOGFIRE_TOKEN")
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT")

    MAX_PAGE_PER_PARSE: int = int(os.getenv("MAX_PAGE_PER_PARSE", "20"))

    model_config = SettingsConfigDict(env_file="../.env")


config = Config()