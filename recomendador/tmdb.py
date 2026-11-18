"""Cliente mínimo da API da TMDB (https://developer.themoviedb.org/)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests

BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"


@dataclass
class Movie:
    id: int
    title: str
    overview: str
    genres: list[str] = field(default_factory=list)
    release_date: str | None = None
    poster_url: str | None = None
    popularity: float = 0.0
    vote_average: float = 0.0

    @property
    def year(self) -> str:
        return (self.release_date or "")[:4]


@dataclass
class Review:
    author: str
    content: str
    rating: float | None = None


class TmdbError(RuntimeError):
    pass


class TmdbClient:
    """Aceita a chave v3 (api_key) ou o token de leitura v4 (Bearer)."""

    def __init__(
        self,
        token: str,
        language: str = "pt-BR",
        region: str = "BR",
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not token:
            raise TmdbError("TMDB_TOKEN não configurado (veja o arquivo .env.example).")
        self._token = token
        self.language = language
        self.region = region
        self._session = session or requests.Session()
        self._timeout = timeout
        self._genre_names: dict[int, str] | None = None

    # ------------------------------------------------------------------ http
    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        headers = {"accept": "application/json"}
        # Tokens v4 são JWTs longos; chaves v3 têm 32 caracteres.
        if len(self._token) > 40:
            headers["Authorization"] = f"Bearer {self._token}"
        else:
            params["api_key"] = self._token
        params = {k: v for k, v in params.items() if v is not None}
        resp = self._session.get(
            f"{BASE_URL}{path}", params=params, headers=headers, timeout=self._timeout
        )
        if resp.status_code != 200:
            raise TmdbError(f"TMDB {path} respondeu {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    # --------------------------------------------------------------- parsing
    def genre_names(self) -> dict[int, str]:
        if self._genre_names is None:
            data = self._get("/genre/movie/list", language=self.language)
            self._genre_names = {g["id"]: g["name"] for g in data.get("genres", [])}
        return self._genre_names

    def _to_movie(self, raw: dict[str, Any]) -> Movie:
        if "genres" in raw:  # resposta de /movie/{id}
            genres = [g["name"] for g in raw["genres"]]
        else:  # resultados de listas trazem só os ids
            names = self.genre_names()
            genres = [names[g] for g in raw.get("genre_ids", []) if g in names]
        poster = raw.get("poster_path")
        return Movie(
            id=raw["id"],
            title=raw.get("title") or raw.get("original_title") or "",
            overview=raw.get("overview") or "",
            genres=genres,
            release_date=raw.get("release_date") or None,
            poster_url=f"{POSTER_BASE}{poster}" if poster else None,
            popularity=float(raw.get("popularity") or 0.0),
            vote_average=float(raw.get("vote_average") or 0.0),
        )

    def _list(self, path: str, **params: Any) -> list[Movie]:
        data = self._get(path, language=self.language, **params)
        return [self._to_movie(r) for r in data.get("results", [])]

    # ------------------------------------------------------------ endpoints
    def search(self, query: str, page: int = 1) -> list[Movie]:
        return self._list("/search/movie", query=query, page=page, include_adult="false")

    def details(self, movie_id: int) -> Movie:
        return self._to_movie(self._get(f"/movie/{movie_id}", language=self.language))

    def upcoming(self, page: int = 1) -> list[Movie]:
        return self._list("/movie/upcoming", page=page, region=self.region)

    def discover(self, genre_ids: list[int], page: int = 1, min_votes: int = 50) -> list[Movie]:
        return self._list(
            "/discover/movie",
            # "|" = OU: filmes de qualquer um dos gêneros
            with_genres="|".join(str(g) for g in genre_ids),
            page=page,
            sort_by="vote_average.desc",
            **{"vote_count.gte": min_votes},
        )

    def recommendations(self, movie_id: int, page: int = 1) -> list[Movie]:
        return self._list(f"/movie/{movie_id}/recommendations", page=page)

    def reviews(self, movie_id: int, page: int = 1) -> list[Review]:
        # Sem filtro de idioma: a maioria das avaliações está em inglês, e tudo bem.
        data = self._get(f"/movie/{movie_id}/reviews", page=page)
        return [
            Review(
                author=r.get("author", ""),
                content=r.get("content", ""),
                rating=(r.get("author_details") or {}).get("rating"),
            )
            for r in data.get("results", [])
        ]

    def genre_ids_for(self, names: list[str]) -> list[int]:
        by_name = {v: k for k, v in self.genre_names().items()}
        return [by_name[n] for n in names if n in by_name]
