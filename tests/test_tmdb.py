from __future__ import annotations

import pytest

from recomendador.tmdb import POSTER_BASE, TmdbClient, TmdbError

GENRES = {"genres": [{"id": 27, "name": "Terror"}, {"id": 878, "name": "Ficção científica"}]}


class FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status
        self.text = str(data)

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        for suffix, data in self.routes.items():
            if url.endswith(suffix):
                return FakeResponse(data)
        return FakeResponse({"status_message": "not found"}, status=404)


def client(routes, token="a" * 32):
    return TmdbClient(token, session=FakeSession(routes))


def test_token_vazio_gera_erro_claro():
    with pytest.raises(TmdbError):
        TmdbClient("")


def test_chave_v3_vai_como_parametro_e_token_v4_como_bearer():
    c3 = client({"/genre/movie/list": GENRES})
    c3.genre_names()
    _, params, headers = c3._session.calls[0]
    assert params["api_key"] == "a" * 32 and "Authorization" not in headers

    c4 = client({"/genre/movie/list": GENRES}, token="x" * 200)
    c4.genre_names()
    _, params, headers = c4._session.calls[0]
    assert headers["Authorization"] == "Bearer " + "x" * 200 and "api_key" not in params


def test_busca_converte_ids_de_genero_e_poster():
    c = client({
        "/genre/movie/list": GENRES,
        "/search/movie": {"results": [{
            "id": 348, "title": "Alien, o Oitavo Passageiro", "overview": "Uma tripulação...",
            "genre_ids": [27, 878, 999], "release_date": "1979-05-25",
            "poster_path": "/p.jpg", "popularity": 50.5, "vote_average": 8.2,
        }]},
    })
    [m] = c.search("alien")
    assert m.genres == ["Terror", "Ficção científica"]  # id desconhecido é ignorado
    assert m.poster_url == f"{POSTER_BASE}/p.jpg"
    assert m.year == "1979"


def test_detalhes_usam_nomes_de_genero_da_resposta():
    c = client({"/movie/348": {"id": 348, "title": "Alien", "overview": "x",
                               "genres": [{"id": 27, "name": "Terror"}], "poster_path": None}})
    m = c.details(348)
    assert m.genres == ["Terror"] and m.poster_url is None


def test_reviews_sem_filtro_de_idioma():
    c = client({"/movie/348/reviews": {"results": [
        {"author": "fan", "content": "The alien is clearly a guy in a suit",
         "author_details": {"rating": 4.0}},
    ]}})
    [r] = c.reviews(348)
    assert r.rating == 4.0 and "guy in a suit" in r.content
    _, params, _ = c._session.calls[0]
    assert "language" not in params


def test_erro_http_vira_tmdberror():
    with pytest.raises(TmdbError):
        client({}).details(1)
