from functools import lru_cache
from typing import Annotated

from utils.config import Config

from fastapi import FastAPI, Depends

app = FastAPI(title="Advanced Rag")


@lru_cache()
def get_config():
    return Config()


@app.get("/")
def home(config: Annotated[Config, Depends(get_config)]):
    return "Running..."
