"""Configuração lida de variáveis de ambiente (arquivo .env em desenvolvimento)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    tmdb_token: str
    database_url: str
    embedding_model: str
    embedding_dim: int
    language: str
    region: str
    gemini_api_key: str
    gemini_models: list[str]


def get_settings() -> Settings:
    return Settings(
        tmdb_token=os.getenv("TMDB_TOKEN", ""),
        database_url=os.getenv(
            "DATABASE_URL", "postgresql://filmes:filmes@localhost:5433/filmes"
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
        language=os.getenv("TMDB_LANGUAGE", "pt-BR"),
        region=os.getenv("TMDB_REGION", "BR"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        # Primeiro o mais rápido e barato; os outros são reserva se ele falhar.
        gemini_models=[
            m.strip()
            for m in os.getenv(
                "GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-4-hydrogen"
            ).split(",")
            if m.strip()
        ],
    )
