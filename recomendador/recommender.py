"""Lógica de recomendação (V1): perfil por embeddings + controle de "descoberta".

A etapa de LLM (reordenar e explicar) entra na próxima iteração, sobre os
candidatos que saem daqui.
"""

from __future__ import annotations

import math

import numpy as np

from .db import Candidate, MovieStore
from .embeddings import Embedder
from .tmdb import Movie, TmdbClient


def build_profile(liked: np.ndarray) -> np.ndarray:
    """Média normalizada dos vetores dos filmes curtidos."""
    if len(liked) == 0:
        raise ValueError("Escolha pelo menos um filme.")
    mean = np.asarray(liked, dtype=np.float32).mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


def apply_discovery(candidates: list[Candidate], discovery: float, k: int) -> list[Candidate]:
    """Reordena penalizando filmes populares.

    discovery = 0 → só similaridade; discovery = 1 → favorece fortemente
    filmes menos conhecidos. A popularidade da TMDB tem cauda longa, por isso
    usamos log e normalizamos entre os candidatos.
    """
    if not candidates:
        return []
    discovery = min(max(discovery, 0.0), 1.0)
    logs = [math.log1p(max(c.movie.popularity, 0.0)) for c in candidates]
    top = max(logs) or 1.0

    def score(item: tuple[Candidate, float]) -> float:
        cand, log_pop = item
        return cand.similarity - 0.3 * discovery * (log_pop / top)

    ranked = sorted(zip(candidates, logs), key=score, reverse=True)
    return [c for c, _ in ranked[:k]]


class Recommender:
    def __init__(self, tmdb: TmdbClient, store: MovieStore, embedder: Embedder) -> None:
        self.tmdb = tmdb
        self.store = store
        self.embedder = embedder

    def _index(self, movies: list[Movie]) -> None:
        """Gera embeddings só para filmes ainda não indexados."""
        unique = {m.id: m for m in movies if m.overview}
        if not unique:
            return
        missing = [m for mid, m in unique.items() if mid not in self.store.known_ids(list(unique))]
        if missing:
            self.store.upsert(missing, self.embedder.embed_movies(missing))

    def _candidate_pool(self, liked: list[Movie], pages: int = 2) -> list[Movie]:
        pool: list[Movie] = []
        for m in liked:
            pool += self.tmdb.recommendations(m.id)
        genre_ids = self.tmdb.genre_ids_for(sorted({g for m in liked for g in m.genres}))
        if genre_ids:
            for page in range(1, pages + 1):
                pool += self.tmdb.discover(genre_ids, page=page)
        pool += self.tmdb.upcoming()  # lançamentos também entram na disputa
        return pool

    def recommend(self, liked_ids: list[int], k: int = 10, discovery: float = 0.3) -> list[Candidate]:
        liked = [self.tmdb.details(i) for i in liked_ids]
        self._index(liked)
        self._index(self._candidate_pool(liked))
        profile = build_profile(self.store.embeddings_for(liked_ids))
        candidates = self.store.nearest(profile, exclude=liked_ids, limit=k * 4)
        return apply_discovery(candidates, discovery, k)
