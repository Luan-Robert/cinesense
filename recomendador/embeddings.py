"""Embeddings de sinopses com um modelo multilíngue do Hugging Face."""

from __future__ import annotations

import numpy as np

from .tmdb import Movie


def movie_text(movie: Movie) -> str:
    genres = ", ".join(movie.genres)
    return f"{movie.title}. Gêneros: {genres}. {movie.overview}".strip()


class Embedder:
    """Carrega o modelo só no primeiro uso (o download leva alguns segundos)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_movies(self, movies: list[Movie]) -> np.ndarray:
        # Modelos E5 esperam o prefixo "passage: " para documentos.
        texts = [f"passage: {movie_text(m)}" for m in movies]
        return self._load().encode(texts, normalize_embeddings=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self._load().encode([f"query: {text}"], normalize_embeddings=True)[0]
