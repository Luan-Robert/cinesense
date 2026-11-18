"""Armazenamento de filmes e vetores no PostgreSQL com a extensão pgvector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .tmdb import Movie

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS movies (
    id            INTEGER PRIMARY KEY,
    title         TEXT NOT NULL,
    overview      TEXT NOT NULL DEFAULT '',
    genres        TEXT[] NOT NULL DEFAULT '{{}}',
    release_date  DATE,
    poster_url    TEXT,
    popularity    REAL NOT NULL DEFAULT 0,
    vote_average  REAL NOT NULL DEFAULT 0,
    embedding     vector({dim}),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- V2: resumo por aspectos da opinião do público, gerado uma vez e reaproveitado.
CREATE TABLE IF NOT EXISTS review_summaries (
    movie_id    INTEGER PRIMARY KEY REFERENCES movies(id) ON DELETE CASCADE,
    summary     JSONB NOT NULL,
    source      TEXT NOT NULL,          -- 'tmdb' ou 'web'
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def as_array(value) -> np.ndarray:
    """Converte o valor de uma coluna vector em array.

    Versões recentes do pgvector devolvem um objeto Vector, não uma lista
    (bug encontrado no primeiro teste com dados reais).
    """
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)


@dataclass
class Candidate:
    movie: Movie
    similarity: float


class MovieStore:
    def __init__(self, database_url: str, dim: int) -> None:
        self.conn = psycopg.connect(database_url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(SCHEMA.format(dim=dim))

    def known_ids(self, ids: list[int]) -> set[int]:
        rows = self.conn.execute(
            "SELECT id FROM movies WHERE id = ANY(%s) AND embedding IS NOT NULL", (ids,)
        ).fetchall()
        return {r[0] for r in rows}

    def upsert(self, movies: list[Movie], embeddings: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            for m, emb in zip(movies, embeddings):
                cur.execute(
                    """
                    INSERT INTO movies (id, title, overview, genres, release_date,
                                        poster_url, popularity, vote_average, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title, overview = EXCLUDED.overview,
                        genres = EXCLUDED.genres, release_date = EXCLUDED.release_date,
                        poster_url = EXCLUDED.poster_url, popularity = EXCLUDED.popularity,
                        vote_average = EXCLUDED.vote_average, embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    (m.id, m.title, m.overview, m.genres, m.release_date, m.poster_url,
                     m.popularity, m.vote_average, np.asarray(emb, dtype=np.float32)),
                )

    def embeddings_for(self, ids: list[int]) -> np.ndarray:
        rows = self.conn.execute(
            "SELECT embedding FROM movies WHERE id = ANY(%s) AND embedding IS NOT NULL",
            (ids,),
        ).fetchall()
        return np.array([as_array(r[0]) for r in rows], dtype=np.float32)

    def nearest(self, profile: np.ndarray, exclude: list[int], limit: int) -> list[Candidate]:
        # <=> é a distância de cosseno do pgvector; similaridade = 1 - distância.
        rows = self.conn.execute(
            """
            SELECT id, title, overview, genres, release_date::text, poster_url,
                   popularity, vote_average, 1 - (embedding <=> %s) AS similarity
            FROM movies
            WHERE embedding IS NOT NULL AND NOT (id = ANY(%s))
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (np.asarray(profile, dtype=np.float32), exclude,
             np.asarray(profile, dtype=np.float32), limit),
        ).fetchall()
        return [
            Candidate(
                movie=Movie(id=r[0], title=r[1], overview=r[2], genres=list(r[3] or []),
                            release_date=r[4], poster_url=r[5], popularity=r[6],
                            vote_average=r[7]),
                similarity=float(r[8]),
            )
            for r in rows
        ]
