from __future__ import annotations

import numpy as np
import pytest

from recomendador.db import Candidate
from recomendador.recommender import apply_discovery, build_profile
from recomendador.tmdb import Movie


def cand(mid, sim, pop):
    return Candidate(Movie(id=mid, title=str(mid), overview="", popularity=pop), sim)


def test_perfil_e_media_normalizada():
    p = build_profile(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(p, [2 ** -0.5, 2 ** -0.5])


def test_perfil_vazio_gera_erro():
    with pytest.raises(ValueError):
        build_profile(np.empty((0, 3)))


def test_sem_descoberta_ordena_so_por_similaridade():
    cands = [cand(1, 0.80, 1000), cand(2, 0.78, 1)]
    assert [c.movie.id for c in apply_discovery(cands, 0.0, 2)] == [1, 2]


def test_descoberta_alta_favorece_filme_desconhecido():
    cands = [cand(1, 0.80, 1000), cand(2, 0.78, 1)]
    assert [c.movie.id for c in apply_discovery(cands, 1.0, 2)] == [2, 1]


def test_descoberta_respeita_k_e_lista_vazia():
    assert apply_discovery([], 0.5, 3) == []
    assert len(apply_discovery([cand(i, 0.5, i) for i in range(10)], 0.5, 3)) == 3
